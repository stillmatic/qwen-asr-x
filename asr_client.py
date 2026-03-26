"""HTTP client for the vLLM-served Qwen3-ASR model."""

import base64
import io
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
import soundfile as sf
from qwen_asr import parse_asr_output

SAMPLE_RATE = 16000
DEFAULT_URL = "http://localhost:8000"


def _audio_to_data_url(audio: np.ndarray, sr: int = SAMPLE_RATE) -> str:
    """Encode numpy audio as a base64 data URL (WAV format)."""
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:audio/wav;base64,{b64}"


def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr)


def is_server_ready(base_url: str = DEFAULT_URL) -> bool:
    try:
        r = requests.get(f"{base_url}/health", timeout=2)
        return r.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


def start_server(
    model: str = "Qwen/Qwen3-ASR-1.7B",
    base_url: str = DEFAULT_URL,
    extra_args: list[str] | None = None,
) -> subprocess.Popen:
    """Start qwen-asr-serve if not already running. Returns the process handle."""
    if is_server_ready(base_url):
        _log(f"vLLM server already running at {base_url}")
        return None

    port = base_url.rsplit(":", 1)[-1].split("/")[0]
    cmd = ["uv", "run", "qwen-asr-serve", model, "--port", port]
    if extra_args:
        cmd.extend(extra_args)

    _log(f"Starting vLLM server: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to be ready
    deadline = time.time() + 300  # 5 min timeout for model loading
    while time.time() < deadline:
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            raise RuntimeError(f"vLLM server exited with code {proc.returncode}:\n{stderr}")
        if is_server_ready(base_url):
            _log("vLLM server is ready")
            return proc
        time.sleep(2)

    proc.terminate()
    raise TimeoutError("vLLM server did not become ready within 5 minutes")


def transcribe_segment(
    audio: np.ndarray,
    *,
    base_url: str = DEFAULT_URL,
    language: str | None = None,
) -> tuple[str, str]:
    """Transcribe a single audio segment via the vLLM server.

    Returns (language, text).
    """
    data_url = _audio_to_data_url(audio)

    data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": data_url}},
                ],
            }
        ],
    }

    response = requests.post(
        f"{base_url}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=data,
        timeout=600,
    )
    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]
    lang, text = parse_asr_output(content, user_language=language)
    return lang, text


def transcribe_segments_concurrent(
    audio_segments: list[np.ndarray],
    *,
    base_url: str = DEFAULT_URL,
    language: str | None = None,
    max_workers: int = 8,
) -> list[tuple[str, str]]:
    """Transcribe multiple segments concurrently via the vLLM server.

    vLLM handles batching internally — sending concurrent requests
    is the most efficient way to utilize it.
    """
    results = [None] * len(audio_segments)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {}
        for i, seg_audio in enumerate(audio_segments):
            future = pool.submit(
                transcribe_segment,
                seg_audio,
                base_url=base_url,
                language=language,
            )
            future_to_idx[future] = i

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()

    return results
