# Features & Performance

## Parallel Pipeline (Multi-VAD + Batched ASR)

The server uses a multi-stage architecture with parallel VAD and batched ASR:

```
                  ┌─ [VAD worker 1 (CPU)] ─┐
job_queue ────────┼─ [VAD worker 2 (CPU)] ─┼──── asr_queue ──── [ASR batcher (GPU)]
                  └─ [VAD worker 3 (CPU)] ─┘
```

- **N VAD workers** (`--vad-workers`, default 2): each loads its own Silero model copy and processes jobs concurrently on CPU. ~12s per 21-min file.
- **ASR batcher**: drains all ready jobs from the queue and merges their segments into a single ASR call. Segments from multiple jobs are transcribed and aligned in one GPU batch.

This design exploits the CPU/GPU split: multiple files go through VAD in parallel on CPU, and the ASR batcher keeps the GPU fed with large, efficient batches.

### Benchmark Results

RTX 4090, Qwen3-ASR-1.7B, no alignment, `examples/npr.mp3` (21 min) for every job, 3 VAD workers:

| Jobs | Wall time | Throughput | Max queue wait |
|------|-----------|-----------|----------------|
| 1    | 32s       | 40x RT    | 0s             |
| 5    | 86s       | 74x RT    | 12s            |
| 10   | 156s      | 81x RT    | 38s            |

For comparison, a naive serial pipeline would take ~320s for 10 jobs (10 x 32s).

### Observations

- **VAD time is stable** (~12s per file) regardless of load. Each worker has its own Silero model copy, no contention.
- **Batching helps throughput**: the ASR batcher merges segments from 2-3 jobs into single GPU calls, keeping utilization high.
- **Queue wait is low**: with 3 VAD workers, at most 3 jobs process VAD simultaneously. Max queue wait for 10 jobs is ~38s vs ~166s without parallelism.
- **Throughput scales well**: 40x realtime (1 job) to 81x realtime (10 jobs). The GPU is the bottleneck, and batching amortizes per-request overhead.

## Benchmarking

```bash
# Start server
uv run python server.py

# Run benchmark
uv run python bench.py --jobs 5
uv run python bench.py --jobs 10 --no-align --json
uv run python bench.py --audio-dir /path/to/files/ --jobs 8
```

The benchmark submits all jobs concurrently, polls until completion, and reports per-job timing (queue wait, VAD time, ASR time) plus aggregate throughput.

## Possible Future Improvements

### Audio pre-loading
Audio decoding (ffmpeg, ~3s for 21min) is I/O-bound and currently part of the VAD stage. A separate pre-load step could decode the next job's audio while the current job's VAD runs, shaving ~3s per job.

### vLLM tuning
Tuning vLLM settings could improve ASR throughput under load:
- `--max-num-batched-tokens`: controls chunked prefill size
- `--gpu-memory-utilization`: trading KV cache for model memory
- `--max-model-len`: capping sequence length to reduce KV cache pressure

### Streaming intra-job overlap
For very long files, the pipeline could split audio into chunks (e.g. 5 minutes), run VAD per-chunk, and start ASR on the first chunk while VAD processes later chunks. This would reduce first-result latency for long files at the cost of slightly less accurate VAD at chunk boundaries.
