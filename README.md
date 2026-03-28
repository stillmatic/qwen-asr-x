# qwen-asr-x

Experimenting with continuous batching strategies for speech transcription. Uses [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) (via vLLM) and [Silero VAD](https://github.com/snakers4/silero-vad), inspired by [WhisperX](https://github.com/m-bain/whisperx).

Takes any audio format (via ffmpeg) and produces transcripts with word-level timestamps, automatic language detection, and optional speaker diarization.

The server implements three key optimizations for multi-job throughput:

1. **VAD/ASR disaggregation** -- VAD (CPU-bound) and ASR (GPU-bound) run as separate pipeline stages, so the next job's VAD overlaps with the current job's ASR.
2. **Parallel VAD workers** -- N copies of Silero VAD process incoming jobs concurrently on CPU, keeping the GPU fed.
3. **Cross-job ASR batching** -- the ASR worker drains all ready jobs and merges their segments into a single vLLM call, improving GPU utilization.

Result: 81x realtime throughput for 10 concurrent jobs on an RTX 4090 (vs 40x for a single job). See [docs/FEATURES.md](docs/FEATURES.md) for benchmark details.

## Pipeline

```
Audio file (mp3, wav, m4a, ...)
    │
    ▼
┌──────────────────┐
│  ffmpeg / librosa │  Load + resample to 16kHz mono
└────────┬─────────┘
         ▼
┌──────────────────┐
│    Silero VAD     │  Detect speech segments, filter silence
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Qwen3-ASR-1.7B  │  Transcribe each segment (vLLM backend)
│     (vLLM)       │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Qwen3-ForcedAlign│  Word-level timestamps via forced alignment
│      0.6B        │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ pyannote speaker │  (Optional) Assign speaker labels
│  diarization     │
└────────┬─────────┘
         ▼
    JSON or SRT output
```

## Requirements

- Python 3.13+
- CUDA GPU (tested on NVIDIA GPUs with 16GB+ VRAM)
- ffmpeg (for audio format support)
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

```bash
# Clone and install dependencies
git clone <repo-url> && cd qwen-asr-x
uv sync

# Models are downloaded automatically on first run from HuggingFace
```

## CLI Usage

```bash
# Basic transcription with word timestamps → JSON
uv run python main.py audio.mp3 -o output.json

# SRT subtitles
uv run python main.py audio.mp3 -o output.srt

# Skip forced alignment (faster, no word timestamps)
uv run python main.py audio.mp3 -o output.json --no-align

# With speaker diarization
uv run python main.py audio.mp3 -o output.json --diarize

# Smaller/faster model
uv run python main.py audio.mp3 -o output.json --model Qwen/Qwen3-ASR-0.6B

# Force language (skip auto-detection)
uv run python main.py audio.mp3 -o output.json --language English
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `audio` | (required) | Input audio file (any ffmpeg format) |
| `-o, --output` | (required) | Output path (.json or .srt) |
| `--model` | `Qwen/Qwen3-ASR-1.7B` | ASR model |
| `--aligner` | `Qwen/Qwen3-ForcedAligner-0.6B` | Forced aligner model |
| `--no-align` | false | Skip forced alignment |
| `--device` | `cuda:0` | GPU device |
| `--language` | auto | Force language |
| `--batch-size` | 4 | Max inference batch size |
| `--gpu-memory-utilization` | 0.8 | vLLM GPU memory target |
| `--diarize` | false | Enable speaker diarization |
| `--min-speakers` | auto | Min speakers hint |
| `--max-speakers` | auto | Max speakers hint |
| `--hf-token` | `$HF_TOKEN` | HuggingFace token (for gated models) |

## Server Mode

For production use, the server loads all models once at startup and processes jobs via an async queue. See [docs/SERVER_INTEGRATION.md](docs/SERVER_INTEGRATION.md) for the full API reference and integration guide.

```bash
# Start the server (loads models once, ~50s)
uv run python server.py

# Submit a job
curl -X POST http://localhost:9090/transcribe \
  -H "Content-Type: application/json" \
  -d '{"audio_path": "/path/to/audio.mp3"}'

# Poll for result
curl http://localhost:9090/jobs/<job_id>
```

## Output Formats

### JSON

```json
[
  {
    "start": 0.22,
    "end": 16.74,
    "text": "This message comes from Allianz Travel Insurance.",
    "language": "English",
    "speaker": "SPEAKER_00",
    "words": [
      {"word": "This", "start": 0.3, "end": 0.54, "speaker": "SPEAKER_00"},
      {"word": "message", "start": 0.54, "end": 1.1, "speaker": "SPEAKER_00"}
    ]
  }
]
```

Fields `speaker` and `words[].speaker` are `null` when diarization is not enabled. The `words` array is empty when `--no-align` is used.

### SRT

```
1
00:00:00,220 --> 00:00:16,740
[SPEAKER_00] This message comes from Allianz Travel Insurance.
```

Speaker prefixes are omitted when diarization is not enabled.

## Performance

Benchmarked on a 21-minute NPR podcast, RTX 4090, Qwen3-ASR-1.7B, warm server:

| Concurrent jobs | Wall time | Throughput |
|-----------------|-----------|-----------|
| 1               | 32s       | 40x RT    |
| 5               | 86s       | 74x RT    |
| 10              | 156s      | 81x RT    |

Single-job breakdown: ~12s VAD (CPU) + ~20s ASR (GPU). Under load, parallel VAD workers and cross-job ASR batching keep the GPU saturated. See [docs/FEATURES.md](docs/FEATURES.md) for full benchmark results and architecture details.

## Project Structure

```
main.py          CLI entrypoint
server.py        FastAPI server with parallel VAD workers + ASR batcher
pipeline.py      Pipeline orchestration (prepare/run_asr_batch stages)
vad.py           Silero VAD wrapper
asr_backend.py   Pluggable ASR backends (Qwen vLLM, Cohere)
output.py        JSON/SRT formatters and data structures
diarize.py       pyannote speaker diarization
bench.py         Server throughput benchmark
vad_viewer.py    Interactive VAD debug viewer
```

## Supported Languages

Auto-detected or force via `--language`: Chinese, English, Cantonese, Arabic, German, French, Spanish, Portuguese, Indonesian, Italian, Korean, Russian, Thai, Vietnamese, Japanese, Turkish, Hindi, Malay, Dutch, Swedish, Danish, Finnish, Polish, Czech, Filipino, Persian, Greek, Romanian, Hungarian, Macedonian.
