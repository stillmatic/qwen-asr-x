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
    SpeechSegment,
    detect_speech,
    extract_segment_audio,
    get_speech_probs,
    load_vad_model,
)


@dataclass
class PreparedJob:
    """Output of the VAD stage, input to the ASR stage."""
    audio: np.ndarray
    vad_segments: list[SpeechSegment]
    vad_params: dict
    audio_path: str
    language: Optional[str]
    prompt: Optional[str]
    align: bool
    diarize: bool
    min_speakers: Optional[int]
    max_speakers: Optional[int]


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
    prompt: Optional[str] = None  # ASR context/prefix to guide transcription
    batch_size: int = 4
    gpu_memory_utilization: float = 0.4
    max_model_len: int = 1536
    enforce_eager: bool = False
    # LLM postprocessing
    llm_postprocess: Optional[str] = None  # "fix" or "translate"
    llm_model: str = "Qwen/Qwen3-4B"
    skip_vad: bool = False
    vad_backend: str = "firered"  # "firered" or "silero"
    # VAD params
    vad_threshold: float = 0.2
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 200
    speech_pad_ms: int = 100
    max_speech_duration_s: float = 60.0
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

    # Default ASR resource limits when LLM postprocessor is not competing for VRAM
    _ASR_DEFAULTS_NO_LLM = {"gpu_memory_utilization": 0.6, "max_model_len": 4096}


    def __init__(self, config: PipelineConfig):
        self.config = config

        # If no LLM postprocessor, give ASR more VRAM (unless user explicitly set values)
        if not config.llm_postprocess:
            for attr, val in self._ASR_DEFAULTS_NO_LLM.items():
                if getattr(config, attr, None) == getattr(PipelineConfig, attr):
                    setattr(config, attr, val)

        # Load VAD model
        _log(f"Loading VAD model ({config.vad_backend})...")
        self.vad_model = load_vad_model(config.vad_backend)

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

        # Optionally load LLM postprocessor
        self.llm_postprocessor = None
        if config.llm_postprocess:
            from llm_postprocess import LLMPostprocessor

            _log(f"Loading LLM postprocessor: {config.llm_model}")
            self.llm_postprocessor = LLMPostprocessor(config.llm_model, config.device)

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

    def prepare(
        self,
        audio_path: str,
        *,
        vad_model=None,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        align: Optional[bool] = None,
        diarize: Optional[bool] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> PreparedJob:
        """Stage 1: Load audio and run VAD (CPU-bound).

        vad_model: Optional separate Silero model instance for thread safety.
                   Falls back to self.vad_model if not provided.
        """
        model = vad_model or self.vad_model
        if align is None:
            align = self.config.align
        if diarize is None:
            diarize = self.config.diarize

        _log(f"Loading audio: {audio_path}")
        audio = load_audio(audio_path)
        duration = len(audio) / SAMPLE_RATE
        _log(f"Audio loaded: {duration:.1f}s ({len(audio)} samples)")

        if self.config.skip_vad:
            _log("Skipping VAD, using entire audio as single segment")
            vad_segments = [SpeechSegment(start=0.0, end=duration)]
            vad_params = {}
        else:
            # VAD
            thresh = self.config.vad_threshold
            _log(f"Running VAD (threshold={thresh})...")
            vad_result = detect_speech(
                audio,
                model,
                threshold=thresh,
                min_speech_duration_ms=self.config.min_speech_duration_ms,
                min_silence_duration_ms=self.config.min_silence_duration_ms,
                speech_pad_ms=self.config.speech_pad_ms,
                max_speech_duration_s=self.config.max_speech_duration_s,
                merge_gap_s=self.config.merge_gap_s,
            )
            vad_segments = vad_result.segments
            vad_params = vad_result.params
            _log(f"VAD found {len(vad_segments)} speech segments")

            if self.config.visualize_vad:
                self._save_vad_debug(audio_path, audio, vad_segments, vad_params, vad_model=model)

        return PreparedJob(
            audio=audio,
            vad_segments=vad_segments,
            vad_params=vad_params,
            audio_path=audio_path,
            language=language or self.config.language,
            prompt=prompt or self.config.prompt,
            align=align,
            diarize=diarize,
            min_speakers=min_speakers or self.config.min_speakers,
            max_speakers=max_speakers or self.config.max_speakers,
        )

    def run_asr(self, prepared: PreparedJob) -> list[TranscriptSegment]:
        """Stage 2: ASR + alignment + diarization for a single job (GPU-bound)."""
        return self.run_asr_batch([prepared])[0]

    def run_asr_batch(self, jobs: list[PreparedJob]) -> list[list[TranscriptSegment]]:
        """Stage 2: ASR + alignment + diarization across multiple jobs in one batch.

        Merges segments from all jobs into single ASR and alignment calls for
        better GPU utilization, then demuxes results back per-job.
        """
        # Build merged segment list with per-job boundaries
        all_audio_segments: list[tuple[np.ndarray, int]] = []
        # job_slices[i] = (start_idx, end_idx) into all_audio_segments
        job_slices: list[tuple[int, int]] = []
        for job in jobs:
            start = len(all_audio_segments)
            for seg in job.vad_segments:
                all_audio_segments.append((extract_segment_audio(job.audio, seg), SAMPLE_RATE))
            job_slices.append((start, len(all_audio_segments)))

        total_segs = len(all_audio_segments)
        prompts = {j.prompt for j in jobs if j.prompt}
        prompt_str = f", prompt={next(iter(prompts))!r}" if len(prompts) == 1 else (f", prompts={prompts}" if prompts else "")
        _log(f"Batch ASR: {total_segs} segments from {len(jobs)} jobs{prompt_str}")

        # Group segments by (language, prompt) for ASR calls
        group_to_seg_indices: dict[tuple[Optional[str], Optional[str]], list[int]] = {}
        for job_idx, job in enumerate(jobs):
            seg_start, seg_end = job_slices[job_idx]
            key = (job.language, job.prompt)
            group_to_seg_indices.setdefault(key, []).extend(range(seg_start, seg_end))

        # Run ASR per group, chunked to avoid OOM
        bs = self.config.batch_size
        all_asr_results = [None] * total_segs
        for (lang, prompt), seg_indices in group_to_seg_indices.items():
            for chunk_start in range(0, len(seg_indices), bs):
                chunk_indices = seg_indices[chunk_start:chunk_start + bs]
                chunk_audio = [all_audio_segments[i] for i in chunk_indices]
                chunk_results = self.asr.transcribe(audio=chunk_audio, language=lang, prompt=prompt)
                for si, result in zip(chunk_indices, chunk_results):
                    all_asr_results[si] = result

        # Forced alignment in one batch (aligner handles per-segment language)
        all_align_results = [None] * total_segs
        any_align = any(job.align for job in jobs)
        if any_align and self.aligner:
            to_align = []
            for i in range(total_segs):
                if all_asr_results[i] is None or not all_asr_results[i].text.strip():
                    continue
                job_idx = next(ji for ji, (s, e) in enumerate(job_slices) if s <= i < e)
                if jobs[job_idx].align:
                    to_align.append((i, all_audio_segments[i], all_asr_results[i]))

            if to_align:
                _log(f"Batch align: {len(to_align)} segments")
                for chunk_start in range(0, len(to_align), bs):
                    chunk = to_align[chunk_start:chunk_start + bs]
                    aligned = self.aligner.align(
                        audio=[sa for _, sa, _ in chunk],
                        text=[r.text for _, _, r in chunk],
                        language=[r.language or "English" for _, _, r in chunk],
                    )
                    for (i, _, _), ar in zip(chunk, aligned):
                        all_align_results[i] = ar

        # Diarization runs per-job in background threads
        diarize_executor = None
        diarize_futures: dict[int, object] = {}
        try:
            diarize_jobs = [
                (i, job) for i, job in enumerate(jobs) if job.diarize and self.diar_pipeline
            ]
            if diarize_jobs:
                from diarize import run_diarization

                diarize_executor = ThreadPoolExecutor(max_workers=len(diarize_jobs))
                for i, job in diarize_jobs:
                    def _run(j=job):
                        _log("Running diarization (background)...")
                        turns = run_diarization(
                            j.audio,
                            self.diar_pipeline,
                            min_speakers=j.min_speakers,
                            max_speakers=j.max_speakers,
                        )
                        _log(f"Diarization complete: {len(set(t.speaker for t in turns))} speakers")
                        return turns
                    diarize_futures[i] = diarize_executor.submit(_run)

            # Demux results per job
            per_job_results: list[list[TranscriptSegment]] = []
            for job_idx, job in enumerate(jobs):
                seg_start, seg_end = job_slices[job_idx]
                results: list[TranscriptSegment] = []
                for seg_i, global_i in enumerate(range(seg_start, seg_end)):
                    asr_r = all_asr_results[global_i]
                    if asr_r is None:
                        continue
                    text = asr_r.text.strip()
                    if not text:
                        continue

                    ar = all_align_results[global_i]
                    seg = job.vad_segments[seg_i]
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
                per_job_results.append(results)

            _log(
                f"Batch complete: {len(jobs)} jobs, "
                f"{sum(len(r) for r in per_job_results)} total segments with text"
            )

            # Apply diarization
            if diarize_futures:
                from diarize import assign_speakers

                for i, future in diarize_futures.items():
                    turns = future.result()
                    per_job_results[i] = assign_speakers(per_job_results[i], turns)

            # LLM postprocessing (fix or translate)
            if self.llm_postprocessor and self.config.llm_postprocess:
                for i, results in enumerate(per_job_results):
                    if results:
                        per_job_results[i] = self.llm_postprocessor.process(
                            results, mode=self.config.llm_postprocess
                        )

            return per_job_results
        finally:
            if diarize_executor is not None:
                diarize_executor.shutdown(wait=True)

    def transcribe(
        self,
        audio_path: str,
        *,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        align: Optional[bool] = None,
        diarize: Optional[bool] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> list[TranscriptSegment]:
        """Run VAD -> ASR -> alignment -> diarization on a single audio file."""
        prepared = self.prepare(
            audio_path,
            language=language,
            prompt=prompt,
            align=align,
            diarize=diarize,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        if not prepared.vad_segments:
            _log("No speech detected")
            return []
        return self.run_asr(prepared)


    def _save_vad_debug(self, audio_path: str, audio: np.ndarray, vad_segments, vad_params: dict, *, vad_model=None):
        """Save VAD probabilities and segments to examples/vad/{stem}/."""
        import json
        from pathlib import Path

        model = vad_model or self.vad_model
        stem = Path(audio_path).stem
        out_dir = Path("examples/vad") / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        _log("Computing VAD probabilities for debug...")
        probs = get_speech_probs(audio, model)
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
