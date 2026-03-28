"""Benchmark: submit N concurrent transcription jobs and measure throughput.

Usage:
    uv run python bench.py
    uv run python bench.py --jobs 10 --audio examples/npr.mp3
    uv run python bench.py --audio-dir /path/to/audio/files/
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx


async def submit_and_poll(
    client: httpx.AsyncClient,
    server: str,
    audio_path: str,
    align: bool,
    poll_interval: float,
) -> dict:
    """Submit a job and poll until terminal status. Returns job response dict."""
    resp = await client.post(
        f"{server}/transcribe",
        json={"audio_path": audio_path, "align": align},
    )
    resp.raise_for_status()
    job_id = resp.json()["job_id"]

    while True:
        await asyncio.sleep(poll_interval)
        resp = await client.get(f"{server}/jobs/{job_id}")
        data = resp.json()
        if data["status"] in ("done", "error"):
            return data


def audio_duration_from_result(result: list[dict]) -> float:
    """Estimate audio duration from the last segment's end time."""
    if not result:
        return 0.0
    return max(seg["end"] for seg in result)


def fmt_sec(s: float | None) -> str:
    if s is None:
        return "-"
    return f"{s:.1f}s"


async def run_bench(args):
    # Resolve audio paths
    if args.audio_dir:
        audio_files = sorted(
            str(p)
            for p in Path(args.audio_dir).iterdir()
            if p.is_file() and p.suffix.lower() in (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".webm")
        )
        if not audio_files:
            print(f"No audio files found in {args.audio_dir}", file=sys.stderr)
            sys.exit(1)
    else:
        audio_path = os.path.abspath(args.audio)
        if not os.path.isfile(audio_path):
            print(f"File not found: {audio_path}", file=sys.stderr)
            sys.exit(1)
        audio_files = [audio_path]

    # Round-robin audio files across jobs
    job_audio = [audio_files[i % len(audio_files)] for i in range(args.jobs)]

    print(f"Benchmark: {args.jobs} jobs, server={args.server}, align={not args.no_align}")
    print(f"Audio files: {len(audio_files)} unique, round-robin across jobs")
    print()

    wall_start = time.monotonic()

    async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
        tasks = [
            submit_and_poll(client, args.server, path, not args.no_align, args.poll_interval)
            for path in job_audio
        ]
        results = await asyncio.gather(*tasks)

    wall_end = time.monotonic()
    wall_time = wall_end - wall_start

    # Check for vad_finished_at support (disaggregated server)
    has_vad_timing = any(r.get("vad_finished_at") is not None for r in results)

    # Header
    if has_vad_timing:
        hdr = f"{'job_id':<14} {'audio':<30} {'queue':>7} {'vad':>7} {'asr':>7} {'total':>7} {'segs':>5} {'status':<6}"
    else:
        hdr = f"{'job_id':<14} {'audio':<30} {'queue':>7} {'proc':>7} {'total':>7} {'segs':>5} {'status':<6}"
    print(hdr)
    print("-" * len(hdr))

    total_audio_dur = 0.0
    errors = 0

    for i, r in enumerate(results):
        job_id = r["job_id"]
        audio_name = Path(job_audio[i]).name
        status = r["status"]

        created = r.get("created_at", 0)
        started = r.get("started_at")
        finished = r.get("finished_at")
        vad_finished = r.get("vad_finished_at")

        queue_wait = (started - created) if started else None
        total = (finished - created) if finished else None

        seg_count = len(r.get("result", []))
        audio_dur = audio_duration_from_result(r.get("result", []))
        total_audio_dur += audio_dur

        if status == "error":
            errors += 1

        if has_vad_timing and started and vad_finished and finished:
            vad_time = vad_finished - started
            asr_time = finished - vad_finished
            print(f"{job_id:<14} {audio_name:<30} {fmt_sec(queue_wait):>7} {fmt_sec(vad_time):>7} {fmt_sec(asr_time):>7} {fmt_sec(total):>7} {seg_count:>5} {status:<6}")
        else:
            proc_time = (finished - started) if (started and finished) else None
            print(f"{job_id:<14} {audio_name:<30} {fmt_sec(queue_wait):>7} {fmt_sec(proc_time):>7} {fmt_sec(total):>7} {seg_count:>5} {status:<6}")

    print()
    print(f"Wall time:           {wall_time:.1f}s")
    print(f"Total audio:         {total_audio_dur:.1f}s")
    if wall_time > 0:
        print(f"Throughput:          {total_audio_dur / wall_time:.1f}x realtime")
    print(f"Jobs:                {args.jobs} ({errors} errors)")

    # Per-job processing time stats
    proc_times = []
    queue_waits = []
    for r in results:
        started = r.get("started_at")
        finished = r.get("finished_at")
        created = r.get("created_at", 0)
        if started and finished:
            proc_times.append(finished - started)
        if started:
            queue_waits.append(started - created)

    if proc_times:
        print(f"Avg processing:      {sum(proc_times) / len(proc_times):.1f}s")
    if queue_waits:
        print(f"Max queue wait:      {max(queue_waits):.1f}s")

    if args.json:
        summary = {
            "wall_time": round(wall_time, 2),
            "total_audio_duration": round(total_audio_dur, 2),
            "throughput": round(total_audio_dur / wall_time, 2) if wall_time > 0 else 0,
            "jobs": args.jobs,
            "errors": errors,
            "avg_processing_time": round(sum(proc_times) / len(proc_times), 2) if proc_times else None,
            "max_queue_wait": round(max(queue_waits), 2) if queue_waits else None,
            "per_job": [
                {
                    "job_id": r["job_id"],
                    "audio": Path(job_audio[i]).name,
                    "status": r["status"],
                    "queue_wait": round((r.get("started_at", 0) - r.get("created_at", 0)), 2) if r.get("started_at") else None,
                    "processing_time": round((r.get("finished_at", 0) - r.get("started_at", 0)), 2) if r.get("started_at") and r.get("finished_at") else None,
                    "vad_time": round((r["vad_finished_at"] - r["started_at"]), 2) if r.get("vad_finished_at") and r.get("started_at") else None,
                    "asr_time": round((r["finished_at"] - r["vad_finished_at"]), 2) if r.get("vad_finished_at") and r.get("finished_at") else None,
                    "audio_duration": round(audio_duration_from_result(r.get("result", [])), 2),
                    "segments": len(r.get("result", [])),
                }
                for i, r in enumerate(results)
            ],
        }
        print()
        print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Benchmark transcription server throughput")
    parser.add_argument("--server", default="http://localhost:9090", help="Server URL (default: http://localhost:9090)")
    parser.add_argument("--audio", default="examples/npr.mp3", help="Audio file for all jobs (default: examples/npr.mp3)")
    parser.add_argument("--audio-dir", default=None, help="Directory of audio files (round-robin across jobs)")
    parser.add_argument("--jobs", type=int, default=5, help="Number of concurrent jobs (default: 5)")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Polling interval in seconds (default: 2.0)")
    parser.add_argument("--no-align", action="store_true", help="Submit with align=false")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary")
    args = parser.parse_args()

    asyncio.run(run_bench(args))


if __name__ == "__main__":
    main()
