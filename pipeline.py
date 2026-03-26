import gc
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from asr_backend import create_backend
from output import TranscriptSegment, WordSegment
from vad import (
    SAMPLE_RATE,
    detect_speech,
    extract_segment_audio,
    get_speech_probs,
    load_vad_model,
)


@dataclass
class PipelineConfig:
    backend: str = "qwen"
    model: str = "Qwen/Qwen3-ASR-1.7B"
    aligner: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    device: str = "cuda:0"
    align: bool = True
    diarize: bool = False
    diarize_model: str = "pyannote/speaker-diarization-community-1"
    hf_token: Optional[str] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    language: Optional[str] = None
    batch_size: int = 4
    gpu_memory_utilization: float = 0.8
    # VAD params
    vad_threshold: float = 0.2
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 200
    speech_pad_ms: int = 100
    max_speech_duration_s: float = 30.0
    merge_gap_s: float = 0.3
    visualize_vad: bool = False


def load_audio(path: str) -> np.ndarray:
    """Load audio to mono float32 @ 16kHz, preferring ffmpeg for speed."""
    if shutil.which("ffmpeg"):
        try:
            proc = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-i",
                    path,
                    "-f",
                    "f32le",
                    "-ac",
                    "1",
                    "-ar",
                    str(SAMPLE_RATE),
                    "pipe:1",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            return np.frombuffer(proc.stdout, dtype=np.float32).copy()
        except (OSError, subprocess.CalledProcessError, ValueError) as exc:
            _log(f"ffmpeg audio decode failed for {path!r}; falling back to librosa ({exc})")
    else:
        _log("ffmpeg not found; falling back to librosa for audio decode")

    import librosa

    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)


def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr)


def _free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class Pipeline:
    """Holds loaded models and runs transcription jobs without reloading."""

    def __init__(self, config: PipelineConfig):
        self.config = config

        # Load VAD model
        _log("Loading VAD model...")
        self.vad_model = load_vad_model()

        # Load ASR backend
        _log(f"Loading ASR backend ({config.backend}): {config.model}")
        self.asr = create_backend(config)

        # Load forced aligner (shared across all backends)
        self.aligner = None
        if config.align:
            from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForcedAligner

            _log(f"Loading forced aligner: {config.aligner}")
            aligner_kwargs = {"device_map": config.device, "dtype": torch.bfloat16}
            if config.hf_token:
                aligner_kwargs["token"] = config.hf_token
            self.aligner = Qwen3ForcedAligner.from_pretrained(
                config.aligner, **aligner_kwargs
            )

        # Optionally pre-load diarization
        self.diar_pipeline = None
        if config.diarize:
            from diarize import load_diarization_pipeline

            _log(f"Loading diarization model: {config.diarize_model}")
            self.diar_pipeline = load_diarization_pipeline(
                model=config.diarize_model,
                device=config.device,
                hf_token=config.hf_token,
            )

        _log("All models loaded")

    def transcribe(
        self,
        audio_path: str,
        *,
        language: Optional[str] = None,
        align: Optional[bool] = None,
        diarize: Optional[bool] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> list[TranscriptSegment]:
        """Run VAD -> ASR -> alignment -> diarization on a single audio file."""
        if align is None:
            align = self.config.align
        if diarize is None:
            diarize = self.config.diarize

        _log(f"Loading audio: {audio_path}")
        audio = load_audio(audio_path)
        duration = len(audio) / SAMPLE_RATE
        _log(f"Audio loaded: {duration:.1f}s ({len(audio)} samples)")

        # VAD
        thresh = self.config.vad_threshold
        _log(f"Running VAD (threshold={thresh})...")
        vad_result = detect_speech(
            audio,
            self.vad_model,
            threshold=thresh,
            min_speech_duration_ms=self.config.min_speech_duration_ms,
            min_silence_duration_ms=self.config.min_silence_duration_ms,
            speech_pad_ms=self.config.speech_pad_ms,
            max_speech_duration_s=self.config.max_speech_duration_s,
            merge_gap_s=self.config.merge_gap_s,
        )
        vad_segments = vad_result.segments
        _log(f"VAD found {len(vad_segments)} speech segments")

        if self.config.visualize_vad:
            self._save_vad_debug(audio_path, audio, vad_segments, vad_result.params)

        if not vad_segments:
            _log("No speech detected")
            return []

        # Kick off diarization in background
        diarize_executor = None
        diarize_future = None
        try:
            if diarize and self.diar_pipeline:
                from diarize import run_diarization

                def _run_diarize():
                    _log("Running diarization (background)...")
                    turns = run_diarization(
                        audio,
                        self.diar_pipeline,
                        min_speakers=min_speakers or self.config.min_speakers,
                        max_speakers=max_speakers or self.config.max_speakers,
                    )
                    _log(f"Diarization complete: {len(set(t.speaker for t in turns))} speakers")
                    return turns

                diarize_executor = ThreadPoolExecutor(max_workers=1)
                diarize_future = diarize_executor.submit(_run_diarize)

            # ASR
            segment_audio = [
                (extract_segment_audio(audio, seg), SAMPLE_RATE) for seg in vad_segments
            ]
            _log(f"Transcribing {len(vad_segments)} segments...")
            asr_results = self.asr.transcribe(
                audio=segment_audio,
                language=language or self.config.language,
            )

            # Forced alignment (shared across all backends)
            align_results = [None] * len(asr_results)
            if align and self.aligner:
                to_align = [
                    (i, segment_audio[i], r)
                    for i, r in enumerate(asr_results)
                    if r.text.strip()
                ]
                if to_align:
                    _log(f"Aligning {len(to_align)} segments...")
                    aligned = self.aligner.align(
                        audio=[sa for _, sa, _ in to_align],
                        text=[r.text for _, _, r in to_align],
                        language=[r.language or "English" for _, _, r in to_align],
                    )
                    for (i, _, _), ar in zip(to_align, aligned):
                        align_results[i] = ar

            # Build results
            results: list[TranscriptSegment] = []
            for seg, asr_r, ar in zip(vad_segments, asr_results, align_results):
                text = asr_r.text.strip()
                if not text:
                    continue

                words = []
                if ar is not None:
                    for item in ar:
                        words.append(
                            WordSegment(
                                word=item.text,
                                start=round(item.start_time + seg.start, 3),
                                end=round(item.end_time + seg.start, 3),
                            )
                        )

                results.append(
                    TranscriptSegment(
                        start=round(seg.start, 3),
                        end=round(seg.end, 3),
                        text=text,
                        language=asr_r.language,
                        words=words,
                    )
                )

            _log(f"Transcription complete: {len(results)} segments with text")

            # Wait for diarization
            if diarize_future is not None:
                from diarize import assign_speakers

                turns = diarize_future.result()
                results = assign_speakers(results, turns)

            return results
        finally:
            if diarize_executor is not None:
                diarize_executor.shutdown(wait=True)


    def _save_vad_debug(self, audio_path: str, audio: np.ndarray, vad_segments, vad_params: dict):
        """Save VAD probabilities and segments to examples/vad/{stem}/."""
        import json
        from pathlib import Path

        stem = Path(audio_path).stem
        out_dir = Path("examples/vad") / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        _log("Computing VAD probabilities for debug...")
        probs = get_speech_probs(audio, self.vad_model)
        duration = len(audio) / SAMPLE_RATE
        window_sec = 512 / SAMPLE_RATE

        data = {
            "audio_path": audio_path,
            "duration": round(duration, 3),
            "sample_rate": SAMPLE_RATE,
            "vad_params": vad_params,
            "probs": {
                "window_sec": round(window_sec, 6),
                "values": [round(p, 4) for p in probs],
            },
            "segments": [
                {"start": round(s.start, 3), "end": round(s.end, 3)}
                for s in vad_segments
            ],
        }

        out_file = out_dir / "vad.json"
        with open(out_file, "w") as f:
            json.dump(data, f, indent=2)
        _log(f"VAD debug saved to {out_file}")


def run_pipeline(audio_path: str, config: PipelineConfig) -> list[TranscriptSegment]:
    """One-shot convenience: load models, transcribe, discard models."""
    pipe = Pipeline(config)
    results = pipe.transcribe(audio_path)
    del pipe
    _free_gpu()
    return results
