"""Async transcription server with job queue.

Start:  uv run python server.py
        uv run python server.py --port 9000 --model Qwen/Qwen3-ASR-0.6B
        uv run python server.py --backend whisper
        uv run python server.py --prompt "domain-specific vocabulary hints"

API:
    POST /transcribe          Submit a job
         Body: {"audio_path": str, "language"?: str, "prompt"?: str,
                "align"?: bool, "diarize"?: bool,
                "min_speakers"?: int, "max_speakers"?: int}
         prompt: context/prefix to guide ASR (vocabulary hints, names, etc.)
                 Supported by qwen (context) and whisper (initial prompt).
    GET  /jobs/{job_id}       Poll job status and result
    GET  /jobs                List all jobs
    POST /jobs/{job_id}/cancel  Cancel a queued or in-progress job
    GET  /stats               Server-level metrics (throughput, RTF, utilization)
    DELETE /jobs/{job_id}     Delete a completed/cancelled job
"""

import argparse
import asyncio
import os
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from enum import Enum
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pipeline import Pipeline, PipelineConfig, PreparedJob, _log
from vad import SAMPLE_RATE, load_vad_model


# --- Stats ---


class ServerStats:
    """Accumulates running server-level metrics."""

    def __init__(self, n_vad_workers: int):
        self.start_time = time.time()
        self.n_vad_workers = n_vad_workers
        self.jobs_completed = 0
        self.jobs_errored = 0
        self.total_audio_seconds = 0.0
        self.total_processing_seconds = 0.0
        self.vad_busy_seconds = 0.0
        self.asr_busy_seconds = 0.0
        self.asr_batches = 0
        self.asr_segments_total = 0
        self._last_summary_completed = 0

    @property
    def uptime(self) -> float:
        return time.time() - self.start_time

    @property
    def throughput(self) -> float:
        """Total audio processed / uptime (Nx realtime)."""
        u = self.uptime
        return self.total_audio_seconds / u if u > 0 else 0.0

    @property
    def avg_rtf(self) -> float | None:
        if self.total_audio_seconds > 0:
            return self.total_processing_seconds / self.total_audio_seconds
        return None

    @property
    def asr_utilization(self) -> float:
        u = self.uptime
        return self.asr_busy_seconds / u if u > 0 else 0.0

    @property
    def vad_utilization(self) -> float:
        u = self.uptime
        total_capacity = u * self.n_vad_workers
        return self.vad_busy_seconds / total_capacity if total_capacity > 0 else 0.0

    @property
    def avg_batch_size(self) -> float | None:
        if self.asr_batches > 0:
            return self.asr_segments_total / self.asr_batches
        return None

    def to_dict(self) -> dict:
        return {
            "uptime": round(self.uptime, 1),
            "jobs_completed": self.jobs_completed,
            "jobs_errored": self.jobs_errored,
            "total_audio_seconds": round(self.total_audio_seconds, 1),
            "throughput": round(self.throughput, 2),
            "avg_rtf": round(self.avg_rtf, 3) if self.avg_rtf is not None else None,
            "asr_utilization": round(self.asr_utilization, 3),
            "vad_utilization": round(self.vad_utilization, 3),
            "avg_batch_size": round(self.avg_batch_size, 1) if self.avg_batch_size is not None else None,
            "asr_batches": self.asr_batches,
            "queue_vad": job_queue.qsize() if job_queue else 0,
            "queue_asr": asr_queue.qsize() if asr_queue else 0,
        }


# --- Models ---

class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    done = "done"
    error = "error"
    cancelled = "cancelled"


class Job:
    def __init__(self, job_id: str, audio_path: str, config: dict):
        self.job_id = job_id
        self.audio_path = audio_path
        self.config = config
        self.status = JobStatus.queued
        self.result = None
        self.error = None
        self.created_at = time.time()
        self.started_at = None
        self.vad_finished_at = None
        self.finished_at = None
        self.audio_duration: float | None = None
        self.num_vad_segments: int = 0
        self._prepared: PreparedJob | None = None


class TranscribeRequest(BaseModel):
    audio_path: str
    language: Optional[str] = None
    prompt: Optional[str] = None
    align: bool = True
    diarize: bool = False
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None


# --- State ---

MAX_COMPLETED_JOBS = 200  # evict oldest completed jobs beyond this
MAX_QUEUE_DEPTH = 50  # reject new jobs if VAD queue exceeds this

jobs: dict[str, Job] = {}
job_queue: asyncio.Queue = None
asr_queue: asyncio.Queue = None
pipeline: Pipeline = None
vad_models: list = []
stats: ServerStats = None


def _evict_old_jobs():
    """Remove oldest completed/errored jobs when count exceeds MAX_COMPLETED_JOBS."""
    terminal = [
        (jid, j) for jid, j in jobs.items()
        if j.status in (JobStatus.done, JobStatus.error)
    ]
    if len(terminal) <= MAX_COMPLETED_JOBS:
        return
    terminal.sort(key=lambda x: x[1].finished_at or 0)
    to_remove = len(terminal) - MAX_COMPLETED_JOBS
    for jid, _ in terminal[:to_remove]:
        del jobs[jid]
    if to_remove > 0:
        _log(f"Evicted {to_remove} old jobs (keeping {MAX_COMPLETED_JOBS})")


# --- Worker ---

_COHERE_LANG_MAP = {
    "arabic": "ar", "chinese": "zh", "dutch": "nl", "english": "en",
    "french": "fr", "german": "de", "greek": "el", "italian": "it",
    "japanese": "ja", "korean": "ko", "polish": "pl", "portuguese": "pt",
    "spanish": "es", "vietnamese": "vi",
}


def _prepare_job(job: Job, vad_model):
    """Stage 1: Load audio + VAD (CPU-bound)."""
    job.status = JobStatus.processing
    job.started_at = time.time()
    _log(f"Job {job.job_id}: VAD stage - {job.audio_path}")

    language = job.config.get("language")
    if language and pipeline.config.backend == "cohere":
        language = _COHERE_LANG_MAP.get(language.lower(), language)

    vad_start = time.time()
    job._prepared = pipeline.prepare(
        job.audio_path,
        vad_model=vad_model,
        language=language,
        prompt=job.config.get("prompt"),
        align=job.config.get("align", True),
        diarize=job.config.get("diarize", False),
        min_speakers=job.config.get("min_speakers"),
        max_speakers=job.config.get("max_speakers"),
    )
    job.vad_finished_at = time.time()
    job.audio_duration = len(job._prepared.audio) / SAMPLE_RATE
    job.num_vad_segments = len(job._prepared.vad_segments)
    stats.vad_busy_seconds += job.vad_finished_at - vad_start
    _log(f"Job {job.job_id}: VAD done ({job.num_vad_segments} segments, {job.audio_duration:.1f}s audio)")


def _run_asr_batch(batch: list[Job]):
    """Stage 2: Batched ASR + alignment + diarization (GPU-bound)."""
    job_ids = ", ".join(j.job_id for j in batch)
    total_segs = sum(j.num_vad_segments for j in batch)
    _log(f"ASR batch: {len(batch)} jobs, {total_segs} segments [{job_ids}]")
    prepared = [job._prepared for job in batch]
    asr_start = time.time()
    results_per_job = pipeline.run_asr_batch(prepared)
    asr_elapsed = time.time() - asr_start
    stats.asr_busy_seconds += asr_elapsed
    stats.asr_batches += 1
    stats.asr_segments_total += total_segs
    now = time.time()
    for job, result in zip(batch, results_per_job):
        job.result = [asdict(s) for s in result]
        job.status = JobStatus.done
        job.finished_at = now
        job._prepared = None  # free audio array
        # Update running stats
        proc_time = job.finished_at - job.started_at
        stats.jobs_completed += 1
        stats.total_audio_seconds += job.audio_duration or 0
        stats.total_processing_seconds += proc_time
        # Per-job completion log
        rtf = proc_time / job.audio_duration if job.audio_duration else 0
        vad_t = (job.vad_finished_at - job.started_at) if job.vad_finished_at and job.started_at else 0
        asr_t = (job.finished_at - job.vad_finished_at) if job.vad_finished_at else proc_time
        _log(
            f"Job {job.job_id} done  |  "
            f"audio: {job.audio_duration:.1f}s  rtf: {rtf:.2f}  vad: {vad_t:.1f}s  asr: {asr_t:.1f}s  |  "
            f"throughput: {stats.throughput:.1f}x  asr-util: {stats.asr_utilization:.0%}"
        )


async def _vad_worker(vad_model):
    """Pull jobs from job_queue, run VAD, push to asr_queue."""
    loop = asyncio.get_event_loop()
    while True:
        job = await job_queue.get()
        try:
            if job.status == JobStatus.cancelled:
                _log(f"Job {job.job_id}: skipped (cancelled)")
                job_queue.task_done()
                continue
            await loop.run_in_executor(None, _prepare_job, job, vad_model)
            if not job._prepared.vad_segments:
                # No speech detected — complete immediately
                job.result = []
                job.status = JobStatus.done
                job.finished_at = time.time()
                job._prepared = None
                _log(f"Job {job.job_id}: no speech detected")
            else:
                await asr_queue.put(job)
        except Exception as e:
            job.status = JobStatus.error
            job.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            job.finished_at = time.time()
            job._prepared = None
            stats.jobs_errored += 1
            _log(f"Job {job.job_id} error  |  {type(e).__name__}: {e}")
        finally:
            job_queue.task_done()


async def _asr_worker():
    """Drain asr_queue and run batched ASR across all ready jobs."""
    loop = asyncio.get_event_loop()
    while True:
        # Wait for at least one job
        job = await asr_queue.get()
        batch = [job]
        # Drain any additional ready jobs (non-blocking)
        while not asr_queue.empty():
            try:
                batch.append(asr_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        # Drop cancelled jobs before running ASR
        cancelled = [j for j in batch if j.status == JobStatus.cancelled]
        for j in cancelled:
            j._prepared = None
            _log(f"Job {j.job_id}: skipped ASR (cancelled)")
        batch = [j for j in batch if j.status != JobStatus.cancelled]
        if not batch:
            for _ in cancelled:
                asr_queue.task_done()
            continue
        try:
            await loop.run_in_executor(None, _run_asr_batch, batch)
        except Exception as e:
            tb = traceback.format_exc()
            for j in batch:
                j.status = JobStatus.error
                j.error = f"{type(e).__name__}: {e}\n{tb}"
                j.finished_at = time.time()
                j._prepared = None
                stats.jobs_errored += 1
            _log(f"ASR batch error ({len(batch)} jobs)  |  {type(e).__name__}: {e}")
        finally:
            for _ in batch + cancelled:
                asr_queue.task_done()


# --- App ---

async def _stats_printer():
    """Print a periodic stats summary every 30s when new jobs have completed."""
    while True:
        await asyncio.sleep(30)
        if stats.jobs_completed > stats._last_summary_completed:
            stats._last_summary_completed = stats.jobs_completed
            avg_batch = stats.avg_batch_size
            avg_batch_str = f"{avg_batch:.1f}" if avg_batch is not None else "-"
            avg_rtf = stats.avg_rtf
            avg_rtf_str = f"{avg_rtf:.2f}" if avg_rtf is not None else "-"
            _log(
                f"-- stats --  "
                f"completed: {stats.jobs_completed}  errored: {stats.jobs_errored}  "
                f"audio: {stats.total_audio_seconds:.1f}s  throughput: {stats.throughput:.1f}x  "
                f"avg-rtf: {avg_rtf_str}  asr-util: {stats.asr_utilization:.0%}  "
                f"avg-batch: {avg_batch_str}"
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global job_queue, asr_queue, stats
    n_vad = len(vad_models)
    job_queue = asyncio.Queue()
    asr_queue = asyncio.Queue(maxsize=n_vad)
    stats = ServerStats(n_vad_workers=n_vad)
    for model in vad_models:
        asyncio.create_task(_vad_worker(model))
    asyncio.create_task(_asr_worker())
    asyncio.create_task(_stats_printer())
    _log(f"Workers started ({n_vad} VAD + 1 ASR batcher), ready to accept jobs")
    yield


app = FastAPI(title="qwen-asr-x", lifespan=lifespan)


@app.post("/transcribe")
async def transcribe(req: TranscribeRequest):
    resolved_audio_path = os.path.abspath(req.audio_path)
    if not os.path.isfile(req.audio_path):
        _log(
            "Rejecting /transcribe request: "
            f"audio_path={req.audio_path!r} resolved_path={resolved_audio_path!r} does not exist"
        )
        raise HTTPException(status_code=400, detail=f"File not found: {req.audio_path}")

    if job_queue.qsize() >= MAX_QUEUE_DEPTH:
        raise HTTPException(status_code=503, detail=f"Queue full ({MAX_QUEUE_DEPTH} pending jobs)")

    _evict_old_jobs()

    job_id = uuid.uuid4().hex[:12]
    job = Job(
        job_id=job_id,
        audio_path=req.audio_path,
        config={
            "language": req.language,
            "prompt": req.prompt,
            "align": req.align,
            "diarize": req.diarize,
            "min_speakers": req.min_speakers,
            "max_speakers": req.max_speakers,
        },
    )
    jobs[job_id] = job
    await job_queue.put(job)

    in_flight = sum(1 for j in jobs.values() if j.status == JobStatus.processing)
    _log(
        f"Job {job_id} queued  |  "
        f"queue: {job_queue.qsize()} vad / {asr_queue.qsize()} asr  |  "
        f"in-flight: {in_flight}"
    )

    return {
        "job_id": job_id,
        "status": job.status,
        "queue_size": job_queue.qsize(),
    }


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    resp = {
        "job_id": job.job_id,
        "status": job.status,
        "audio_path": job.audio_path,
        "audio_duration": job.audio_duration,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "vad_finished_at": job.vad_finished_at,
        "finished_at": job.finished_at,
        "num_vad_segments": job.num_vad_segments,
    }
    if job.status == JobStatus.done:
        resp["result"] = job.result
        resp["num_asr_segments"] = len(job.result)
        if job.started_at and job.finished_at and job.audio_duration:
            proc_time = job.finished_at - job.started_at
            resp["rtf"] = round(proc_time / job.audio_duration, 3)
    if job.status == JobStatus.error:
        resp["error"] = job.error
    return resp


@app.get("/jobs")
async def list_jobs():
    return [
        {
            "job_id": j.job_id,
            "status": j.status,
            "audio_path": j.audio_path,
            "created_at": j.created_at,
        }
        for j in jobs.values()
    ]


@app.get("/stats")
async def get_stats():
    d = stats.to_dict()
    d["jobs_queued"] = job_queue.qsize()
    d["jobs_in_flight"] = sum(1 for j in jobs.values() if j.status == JobStatus.processing)
    return d


@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status in (JobStatus.done, JobStatus.error, JobStatus.cancelled):
        raise HTTPException(status_code=409, detail=f"Job already {job.status.value}")
    job.status = JobStatus.cancelled
    job.finished_at = time.time()
    job._prepared = None
    _log(f"Job {job.job_id} cancelled")
    return {"job_id": job_id, "status": "cancelled"}


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status == JobStatus.processing:
        raise HTTPException(status_code=409, detail="Cannot delete a job in progress")
    del jobs[job_id]
    return {"deleted": job_id}


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="qwen-asr-x transcription server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=9090, help="Bind port (default: 9090)")
    parser.add_argument("--backend", choices=["qwen", "cohere", "whisper"], default="qwen", help="ASR backend")
    parser.add_argument("--model", default=None, help="ASR model (default depends on backend)")
    parser.add_argument("--aligner", default="Qwen/Qwen3-ForcedAligner-0.6B", help="Aligner model")
    parser.add_argument("--device", default="cuda:0", help="Device")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4, help="vLLM GPU memory target (qwen only, default: 0.4)")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable CUDA graphs to save ~500MB VRAM (slightly slower)")
    parser.add_argument("--prompt", default=None, help="Default ASR prompt/context to guide transcription (vocabulary, domain hints)")
    parser.add_argument("--max-queue", type=int, default=50, help="Max queued jobs before rejecting (default: 50)")
    parser.add_argument("--hf-token", default=None, help="HF token for gated models")
    parser.add_argument("--diarize", action="store_true", help="Pre-load diarization model")
    parser.add_argument("--skip-vad", action="store_true", help="Skip VAD and feed entire audio as a single segment")
    parser.add_argument("--vad-backend", choices=["silero", "firered"], default="firered", help="VAD backend (default: silero)")
    # VAD tuning
    parser.add_argument("--vad-threshold", type=float, default=0.2, help="VAD threshold (default: 0.2)")
    parser.add_argument("--min-speech-duration-ms", type=int, default=250, help="Min speech duration ms (default: 250)")
    parser.add_argument("--min-silence-duration-ms", type=int, default=200, help="Min silence duration ms (default: 200)")
    parser.add_argument("--speech-pad-ms", type=int, default=100, help="Speech padding ms (default: 100)")
    parser.add_argument("--visualize-vad", action="store_true", help="Save VAD debug data to examples/vad/")
    parser.add_argument("--vad-workers", type=int, default=4, help="Number of parallel VAD workers (default: 4)")
    # LLM postprocessing
    parser.add_argument("--llm-postprocess", choices=["fix", "translate"], default=None,
                        help="LLM postprocessing: 'fix' corrects errors, 'translate' translates to English")
    parser.add_argument("--llm-model", default="Qwen/Qwen3-4B", help="LLM model for postprocessing (default: Qwen/Qwen3-4B)")
    args = parser.parse_args()

    default_models = {
        "qwen": "Qwen/Qwen3-ASR-1.7B",
        "cohere": "CohereLabs/cohere-transcribe-03-2026",
        "whisper": "openai/whisper-large-v3",
    }
    model = args.model or default_models[args.backend]

    global MAX_QUEUE_DEPTH
    MAX_QUEUE_DEPTH = args.max_queue

    config = PipelineConfig(
        backend=args.backend,
        model=model,
        aligner=args.aligner,
        device=args.device,
        batch_size=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        prompt=args.prompt,
        skip_vad=args.skip_vad,
        vad_backend=args.vad_backend,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN"),
        diarize=args.diarize,
        vad_threshold=args.vad_threshold,
        min_speech_duration_ms=args.min_speech_duration_ms,
        min_silence_duration_ms=args.min_silence_duration_ms,
        speech_pad_ms=args.speech_pad_ms,
        visualize_vad=args.visualize_vad,
        llm_postprocess=args.llm_postprocess,
        llm_model=args.llm_model,
    )

    # Load all models once at startup
    global pipeline, vad_models
    _log(f"Initializing pipeline: {model} ({args.backend})")
    pipeline = Pipeline(config)

    # Load extra VAD models for parallel workers (Pipeline already loaded one)
    n_vad = max(1, args.vad_workers)
    _log(f"Loading {n_vad} VAD model(s) for parallel workers...")
    vad_models = [load_vad_model(config.vad_backend) for _ in range(n_vad)]

    _log(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
