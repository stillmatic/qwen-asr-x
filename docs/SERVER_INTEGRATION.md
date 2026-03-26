# Server Integration Guide

This document describes how to integrate with the qwen-asr-x transcription server from any language (Go, Python, TypeScript, etc.).

## Starting the Server

```bash
# Default: loads Qwen3-ASR-1.7B + ForcedAligner on cuda:0, listens on port 9090
uv run python server.py

# Custom configuration
uv run python server.py \
  --model Qwen/Qwen3-ASR-0.6B \
  --port 8080 \
  --gpu-memory-utilization 0.9 \
  --batch-size 8

# With diarization pre-loaded (requires HF_TOKEN for pyannote)
HF_TOKEN=hf_xxx uv run python server.py --diarize
```

### Server Options

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `9090` | Listen port |
| `--model` | `Qwen/Qwen3-ASR-1.7B` | ASR model |
| `--aligner` | `Qwen/Qwen3-ForcedAligner-0.6B` | Forced aligner model |
| `--device` | `cuda:0` | GPU device |
| `--batch-size` | `4` | Max inference batch size |
| `--gpu-memory-utilization` | `0.8` | vLLM GPU memory target |
| `--diarize` | off | Pre-load diarization model |
| `--hf-token` | `$HF_TOKEN` | HuggingFace token |

Startup takes ~50s (model loading, torch.compile, CUDA graph capture). Once ready, the server logs:

```
[HH:MM:SS] Worker started, ready to accept jobs
```

## API Reference

### `POST /transcribe` — Submit a Job

Submit an audio file for transcription. Returns immediately with a job ID.

**Request:**

```json
{
  "audio_path": "/absolute/path/to/audio.mp3",
  "language": null,
  "align": true,
  "diarize": false,
  "min_speakers": null,
  "max_speakers": null
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `audio_path` | string | (required) | Absolute path to audio file on the server's filesystem |
| `language` | string\|null | null | Force language (null = auto-detect) |
| `align` | bool | true | Run forced alignment for word timestamps |
| `diarize` | bool | false | Run speaker diarization (server must be started with `--diarize`) |
| `min_speakers` | int\|null | null | Hint: minimum number of speakers |
| `max_speakers` | int\|null | null | Hint: maximum number of speakers |

**Response (202-like):**

```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "queued",
  "queue_size": 1
}
```

**Errors:**

- `400` — File not found at the given path

---

### `GET /jobs/{job_id}` — Poll Job Status

Completion is indicated by the string field `status` becoming `"done"`.
The transcript payload is returned in `result`. The server does not return `done: true` or `segments: [...]`.

**Response when queued:**

```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "queued",
  "audio_path": "/path/to/audio.mp3",
  "created_at": 1711440000.0,
  "started_at": null,
  "finished_at": null
}
```

**Response when processing:**

```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "processing",
  "audio_path": "/path/to/audio.mp3",
  "created_at": 1711440000.0,
  "started_at": 1711440005.0,
  "finished_at": null
}
```

**Response when done:**

```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "done",
  "audio_path": "/path/to/audio.mp3",
  "created_at": 1711440000.0,
  "started_at": 1711440005.0,
  "finished_at": 1711440079.0,
  "result": [
    {
      "start": 0.22,
      "end": 16.74,
      "text": "Hello world.",
      "language": "English",
      "speaker": null,
      "words": [
        {"word": "Hello", "start": 0.3, "end": 0.8, "speaker": null},
        {"word": "world", "start": 0.9, "end": 1.3, "speaker": null}
      ]
    }
  ]
}
```

**Response on error:**

```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "error",
  "audio_path": "/path/to/audio.mp3",
  "created_at": 1711440000.0,
  "started_at": 1711440005.0,
  "finished_at": 1711440006.0,
  "error": "FileNotFoundError: ..."
}
```

**Errors:**

- `404` — Job ID not found

---

### `GET /jobs` — List All Jobs

Returns an array of job summaries (without results, to keep the response small).

```json
[
  {
    "job_id": "a1b2c3d4e5f6",
    "status": "done",
    "audio_path": "/path/to/audio.mp3",
    "created_at": 1711440000.0
  }
]
```

---

### `DELETE /jobs/{job_id}` — Delete a Job

Removes a completed or errored job from memory. Cannot delete a job that is currently processing.

```json
{"deleted": "a1b2c3d4e5f6"}
```

**Errors:**

- `404` — Job not found
- `409` — Job is currently processing

---

## Job Lifecycle

```
POST /transcribe
       │
       ▼
   ┌────────┐     ┌────────────┐     ┌──────┐
   │ queued  │────▶│ processing │────▶│ done │
   └────────┘     └────────────┘     └──────┘
                        │
                        ▼
                    ┌───────┐
                    │ error │
                    └───────┘
```

Jobs are processed sequentially in FIFO order (one at a time on the GPU). If you submit multiple jobs, they queue up and execute in order.

## Integration Examples

### Go

The polling response maps to `status` and `result`, so your Go struct should use `Status string` and `Result []Segment`.

```go
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

const baseURL = "http://localhost:9090"

type SubmitResponse struct {
	JobID     string `json:"job_id"`
	Status    string `json:"status"`
	QueueSize int    `json:"queue_size"`
}

type Word struct {
	Word    string   `json:"word"`
	Start   float64  `json:"start"`
	End     float64  `json:"end"`
	Speaker *string  `json:"speaker"`
}

type Segment struct {
	Start    float64  `json:"start"`
	End      float64  `json:"end"`
	Text     string   `json:"text"`
	Language string   `json:"language"`
	Speaker  *string  `json:"speaker"`
	Words    []Word   `json:"words"`
}

type JobResponse struct {
	JobID      string    `json:"job_id"`
	Status     string    `json:"status"`
	AudioPath  string    `json:"audio_path"`
	CreatedAt  float64   `json:"created_at"`
	StartedAt  *float64  `json:"started_at"`
	FinishedAt *float64  `json:"finished_at"`
	Result     []Segment `json:"result,omitempty"`
	Error      string    `json:"error,omitempty"`
}

// Submit a transcription job. Returns the job ID.
func submitJob(audioPath string) (string, error) {
	body, _ := json.Marshal(map[string]any{
		"audio_path": audioPath,
		"align":      true,
		"diarize":    false,
	})

	resp, err := http.Post(baseURL+"/transcribe", "application/json", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var result SubmitResponse
	json.NewDecoder(resp.Body).Decode(&result)
	return result.JobID, nil
}

// Poll until the job is done or errored. Returns the full job response.
func waitForJob(jobID string) (*JobResponse, error) {
	for {
		resp, err := http.Get(fmt.Sprintf("%s/jobs/%s", baseURL, jobID))
		if err != nil {
			return nil, err
		}

		var job JobResponse
		json.NewDecoder(resp.Body).Decode(&job)
		resp.Body.Close()

		switch job.Status {
		case "done":
			return &job, nil
		case "error":
			return nil, fmt.Errorf("job failed: %s", job.Error)
		}

		time.Sleep(2 * time.Second)
	}
}

func main() {
	jobID, err := submitJob("/path/to/audio.mp3")
	if err != nil {
		panic(err)
	}
	fmt.Printf("Submitted job: %s\n", jobID)

	job, err := waitForJob(jobID)
	if err != nil {
		panic(err)
	}

	fmt.Printf("Transcribed %d segments\n", len(job.Result))
	for _, seg := range job.Result {
		fmt.Printf("[%.1f-%.1f] %s\n", seg.Start, seg.End, seg.Text)
	}
}
```

### Python

```python
import time
import requests

BASE_URL = "http://localhost:9090"

def transcribe(audio_path: str, diarize: bool = False) -> list[dict]:
    """Submit a job and poll until complete."""
    # Submit
    resp = requests.post(f"{BASE_URL}/transcribe", json={
        "audio_path": audio_path,
        "diarize": diarize,
    })
    resp.raise_for_status()
    job_id = resp.json()["job_id"]
    print(f"Job submitted: {job_id}")

    # Poll
    while True:
        resp = requests.get(f"{BASE_URL}/jobs/{job_id}")
        data = resp.json()

        if data["status"] == "done":
            return data["result"]
        if data["status"] == "error":
            raise RuntimeError(data["error"])

        time.sleep(2)

segments = transcribe("/path/to/audio.mp3")
for seg in segments:
    print(f"[{seg['start']:.1f}-{seg['end']:.1f}] {seg['text']}")
```

### curl

```bash
# Submit
JOB_ID=$(curl -s -X POST http://localhost:9090/transcribe \
  -H "Content-Type: application/json" \
  -d '{"audio_path": "/path/to/audio.mp3"}' | jq -r '.job_id')

echo "Job: $JOB_ID"

# Poll until done
while true; do
  STATUS=$(curl -s "http://localhost:9090/jobs/$JOB_ID" | jq -r '.status')
  echo "Status: $STATUS"
  [ "$STATUS" = "done" ] || [ "$STATUS" = "error" ] && break
  sleep 2
done

# Get result
curl -s "http://localhost:9090/jobs/$JOB_ID" | jq '.result'
```

## Result Schema

Each segment in the `result` array:

| Field | Type | Description |
|-------|------|-------------|
| `start` | float | Segment start time in seconds |
| `end` | float | Segment end time in seconds |
| `text` | string | Transcribed text |
| `language` | string | Detected language (e.g. "English") |
| `speaker` | string\|null | Speaker label (null if diarization disabled) |
| `words` | array | Word-level timestamps (empty if alignment disabled) |

Each word in the `words` array:

| Field | Type | Description |
|-------|------|-------------|
| `word` | string | The word |
| `start` | float | Word start time in seconds |
| `end` | float | Word end time in seconds |
| `speaker` | string\|null | Speaker label for this word |

## Architecture Notes

- **Job queue is in-process** — jobs are stored in memory and lost on restart. For persistence, implement a database-backed queue upstream.
- **One job at a time** — the GPU processes jobs sequentially. Submitting multiple jobs queues them in FIFO order.
- **Models stay warm** — VAD, ASR (vLLM), aligner, and diarization models are loaded once at server startup. Subsequent jobs skip the ~50s initialization.
- **Audio must be on the local filesystem** — the server reads files by path. If your client is remote, upload the file first (e.g. to a shared volume or temp directory) and pass the path.
- **Diarization is optional** — pass `--diarize` at server startup to pre-load the model. Per-job `"diarize": true` only works if the server was started with diarization enabled.
