import gc
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import librosa
import numpy as np
import torch
from qwen_asr import Qwen3ASRModel

from output import TranscriptSegment, WordSegment
from vad import SAMPLE_RATE, detect_speech, extract_segment_audio, load_vad_model


@dataclass
class PipelineConfig:
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
    vad_threshold: float = 0.5
    max_speech_duration_s: float = 30.0
    merge_gap_s: float = 0.3


def load_audio(path: str) -> np.ndarray:
    """Load any audio file to mono float32 @ 16kHz using librosa (ffmpeg backend)."""
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)


def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr)


def _free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_pipeline(audio_path: str, config: PipelineConfig) -> list[TranscriptSegment]:
    """Run the full VAD -> ASR -> alignment -> diarization pipeline."""
    _log(f"Loading audio: {audio_path}")
    audio = load_audio(audio_path)
    duration = len(audio) / SAMPLE_RATE
    _log(f"Audio loaded: {duration:.1f}s ({len(audio)} samples)")

    # VAD
    _log("Loading VAD model...")
    vad_model = load_vad_model()
    _log("Running VAD...")
    vad_segments = detect_speech(
        audio,
        vad_model,
        threshold=config.vad_threshold,
        max_speech_duration_s=config.max_speech_duration_s,
        merge_gap_s=config.merge_gap_s,
    )
    _log(f"VAD found {len(vad_segments)} speech segments")
    del vad_model

    if not vad_segments:
        _log("No speech detected")
        return []

    # Kick off diarization in background thread (runs in parallel with ASR)
    diarize_future = None
    diar_pipeline = None
    if config.diarize:
        from diarize import load_diarization_pipeline, run_diarization

        _log(f"Loading diarization model: {config.diarize_model}")
        diar_pipeline = load_diarization_pipeline(
            model=config.diarize_model,
            device=config.device,
            hf_token=config.hf_token,
        )

        def _run_diarize():
            _log("Running diarization (background)...")
            turns = run_diarization(
                audio,
                diar_pipeline,
                min_speakers=config.min_speakers,
                max_speakers=config.max_speakers,
            )
            _log(f"Diarization complete: {len(set(t.speaker for t in turns))} speakers")
            return turns

        executor = ThreadPoolExecutor(max_workers=1)
        diarize_future = executor.submit(_run_diarize)

    # Load ASR model
    _log(f"Loading ASR model (vLLM): {config.model}")
    aligner = config.aligner if config.align else None
    aligner_kwargs = None
    if aligner:
        aligner_kwargs = {"device_map": config.device, "dtype": torch.bfloat16}
        if config.hf_token:
            aligner_kwargs["token"] = config.hf_token

    asr = Qwen3ASRModel.LLM(
        model=config.model,
        forced_aligner=aligner,
        forced_aligner_kwargs=aligner_kwargs,
        max_inference_batch_size=config.batch_size,
        max_new_tokens=512,
        gpu_memory_utilization=config.gpu_memory_utilization,
        dtype="bfloat16",
        hf_token=config.hf_token,
    )
    _log("ASR model loaded")

    segment_audio = [
        (extract_segment_audio(audio, seg), SAMPLE_RATE) for seg in vad_segments
    ]
    _log(
        f"Transcribing {len(vad_segments)} segments via in-process vLLM "
        f"(max batch {config.batch_size})..."
    )
    transcriptions = asr.transcribe(
        audio=segment_audio,
        language=config.language,
        return_time_stamps=config.align,
    )

    results: list[TranscriptSegment] = []
    for seg, tx in zip(vad_segments, transcriptions):
        text = (tx.text or "").strip()
        if not text:
            continue

        words = []
        if config.align and tx.time_stamps is not None:
            for item in tx.time_stamps:
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
                language=tx.language or "",
                words=words,
            )
        )

    _log(f"Transcription complete: {len(results)} segments with text")

    # Free ASR model
    del asr
    _free_gpu()

    # Wait for diarization and assign speakers
    if diarize_future is not None:
        from diarize import assign_speakers

        turns = diarize_future.result()
        results = assign_speakers(results, turns)
        del diar_pipeline
        _free_gpu()

    return results
