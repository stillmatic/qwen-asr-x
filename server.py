"""Async transcription server with job queue.

Start:  uv run python server.py
        uv run python server.py --port 9000 --model Qwen/Qwen3-ASR-0.6B

API:
    POST /transcribe        Submit a job (JSON body with "audio_path")
    GET  /jobs/{job_id}     Poll job status and result
    GET  /jobs              List all jobs
    DELETE /jobs/{job_id}   Delete a completed job
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
from vad import load_vad_model


# --- Models ---

class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    done = "done"
    error = "error"


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
        self._prepared: PreparedJob | None = None


class TranscribeRequest(BaseModel):
    audio_path: str
    language: Optional[str] = None
    align: bool = True
    diarize: bool = False
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None


# --- State ---

jobs: dict[str, Job] = {}
job_queue: asyncio.Queue = None
asr_queue: asyncio.Queue = None
pipeline: Pipeline = None
vad_models: list = []


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

    job._prepared = pipeline.prepare(
        job.audio_path,
        vad_model=vad_model,
        language=language,
        align=job.config.get("align", True),
        diarize=job.config.get("diarize", False),
        min_speakers=job.config.get("min_speakers"),
        max_speakers=job.config.get("max_speakers"),
    )
    job.vad_finished_at = time.time()
    _log(f"Job {job.job_id}: VAD done ({len(job._prepared.vad_segments)} segments)")


def _run_asr_batch(batch: list[Job]):
    """Stage 2: Batched ASR + alignment + diarization (GPU-bound)."""
    job_ids = ", ".join(j.job_id for j in batch)
    _log(f"ASR batch: {len(batch)} jobs [{job_ids}]")
    prepared = [job._prepared for job in batch]
    results_per_job = pipeline.run_asr_batch(prepared)
    now = time.time()
    for job, result in zip(batch, results_per_job):
        job.result = [asdict(s) for s in result]
        job.status = JobStatus.done
        job.finished_at = now
        job._prepared = None  # free audio array
        _log(f"Job {job.job_id}: done ({len(result)} segments)")


async def _vad_worker(vad_model):
    """Pull jobs from job_queue, run VAD, push to asr_queue."""
    loop = asyncio.get_event_loop()
    while True:
        job = await job_queue.get()
        try:
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
            _log(f"Job {job.job_id}: error in VAD stage - {e}")
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
        try:
            await loop.run_in_executor(None, _run_asr_batch, batch)
        except Exception as e:
            tb = traceback.format_exc()
            for j in batch:
                j.status = JobStatus.error
                j.error = f"{type(e).__name__}: {e}\n{tb}"
                j.finished_at = time.time()
                j._prepared = None
            _log(f"ASR batch error ({len(batch)} jobs): {e}")
        finally:
            for _ in batch:
                asr_queue.task_done()


# --- App ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    global job_queue, asr_queue
    n_vad = len(vad_models)
    job_queue = asyncio.Queue()
    asr_queue = asyncio.Queue(maxsize=n_vad)
    for model in vad_models:
        asyncio.create_task(_vad_worker(model))
    asyncio.create_task(_asr_worker())
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

    job_id = uuid.uuid4().hex[:12]
    job = Job(
        job_id=job_id,
        audio_path=req.audio_path,
        config={
            "language": req.language,
            "align": req.align,
            "diarize": req.diarize,
            "min_speakers": req.min_speakers,
            "max_speakers": req.max_speakers,
        },
    )
    jobs[job_id] = job
    await job_queue.put(job)

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
        "created_at": job.created_at,
        "started_at": job.started_at,
        "vad_finished_at": job.vad_finished_at,
        "finished_at": job.finished_at,
    }
    if job.status == JobStatus.done:
        resp["result"] = job.result
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
    parser.add_argument("--backend", choices=["qwen", "cohere"], default="qwen", help="ASR backend")
    parser.add_argument("--model", default=None, help="ASR model (default depends on backend)")
    parser.add_argument("--aligner", default="Qwen/Qwen3-ForcedAligner-0.6B", help="Aligner model")
    parser.add_argument("--device", default="cuda:0", help="Device")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8, help="vLLM GPU memory target (qwen only)")
    parser.add_argument("--hf-token", default=None, help="HF token for gated models")
    parser.add_argument("--diarize", action="store_true", help="Pre-load diarization model")
    # VAD tuning
    parser.add_argument("--vad-threshold", type=float, default=0.2, help="VAD threshold (default: 0.2)")
    parser.add_argument("--min-speech-duration-ms", type=int, default=250, help="Min speech duration ms (default: 250)")
    parser.add_argument("--min-silence-duration-ms", type=int, default=200, help="Min silence duration ms (default: 200)")
    parser.add_argument("--speech-pad-ms", type=int, default=100, help="Speech padding ms (default: 100)")
    parser.add_argument("--visualize-vad", action="store_true", help="Save VAD debug data to examples/vad/")
    parser.add_argument("--vad-workers", type=int, default=2, help="Number of parallel VAD workers (default: 2)")
    args = parser.parse_args()

    default_models = {
        "qwen": "Qwen/Qwen3-ASR-1.7B",
        "cohere": "CohereLabs/cohere-transcribe-03-2026",
    }
    model = args.model or default_models[args.backend]

    config = PipelineConfig(
        backend=args.backend,
        model=model,
        aligner=args.aligner,
        device=args.device,
        batch_size=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN"),
        diarize=args.diarize,
        vad_threshold=args.vad_threshold,
        min_speech_duration_ms=args.min_speech_duration_ms,
        min_silence_duration_ms=args.min_silence_duration_ms,
        speech_pad_ms=args.speech_pad_ms,
        visualize_vad=args.visualize_vad,
    )

    # Load all models once at startup
    global pipeline, vad_models
    _log(f"Initializing pipeline: {model} ({args.backend})")
    pipeline = Pipeline(config)

    # Load extra VAD models for parallel workers (Pipeline already loaded one)
    n_vad = max(1, args.vad_workers)
    _log(f"Loading {n_vad} VAD model(s) for parallel workers...")
    vad_models = [load_vad_model() for _ in range(n_vad)]

    _log(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
