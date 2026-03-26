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
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from enum import Enum
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pipeline import PipelineConfig, _log, run_pipeline


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
        self.finished_at = None


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
pipeline_config: PipelineConfig = None


# --- Worker ---

def _process_job(job: Job):
    """Run the pipeline for a single job (called from worker thread)."""
    job.status = JobStatus.processing
    job.started_at = time.time()
    _log(f"Job {job.job_id}: processing {job.audio_path}")

    try:
        config = PipelineConfig(
            model=pipeline_config.model,
            aligner=pipeline_config.aligner,
            device=pipeline_config.device,
            align=job.config.get("align", True),
            diarize=job.config.get("diarize", False),
            diarize_model=pipeline_config.diarize_model,
            hf_token=pipeline_config.hf_token,
            min_speakers=job.config.get("min_speakers"),
            max_speakers=job.config.get("max_speakers"),
            language=job.config.get("language"),
            batch_size=pipeline_config.batch_size,
        )
        segments = run_pipeline(job.audio_path, config)
        job.result = [asdict(s) for s in segments]
        job.status = JobStatus.done
        _log(f"Job {job.job_id}: done ({len(segments)} segments)")
    except Exception as e:
        job.status = JobStatus.error
        job.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        _log(f"Job {job.job_id}: error - {e}")
    finally:
        job.finished_at = time.time()


async def _worker():
    """Process jobs from the queue one at a time."""
    loop = asyncio.get_event_loop()
    while True:
        job = await job_queue.get()
        try:
            await loop.run_in_executor(None, _process_job, job)
        except Exception:
            pass
        job_queue.task_done()


# --- App ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    global job_queue
    job_queue = asyncio.Queue()
    asyncio.create_task(_worker())
    _log("Worker started, ready to accept jobs")
    yield


app = FastAPI(title="qwen-asr-x", lifespan=lifespan)


@app.post("/transcribe")
async def transcribe(req: TranscribeRequest):
    if not os.path.isfile(req.audio_path):
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
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B", help="ASR model")
    parser.add_argument("--aligner", default="Qwen/Qwen3-ForcedAligner-0.6B", help="Aligner model")
    parser.add_argument("--device", default="cuda:0", help="Device")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--hf-token", default=None, help="HF token for diarization")
    args = parser.parse_args()

    global pipeline_config
    pipeline_config = PipelineConfig(
        model=args.model,
        aligner=args.aligner,
        device=args.device,
        batch_size=args.batch_size,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN"),
    )

    _log(f"Starting server on {args.host}:{args.port}")
    _log(f"Model: {args.model}, Aligner: {args.aligner}, Device: {args.device}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
