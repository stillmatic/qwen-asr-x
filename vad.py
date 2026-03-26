from dataclasses import dataclass

import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

SAMPLE_RATE = 16000


@dataclass
class SpeechSegment:
    start: float  # seconds
    end: float  # seconds


def load_vad_model():
    return load_silero_vad()


def detect_speech(
    audio: np.ndarray,
    model,
    *,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 200,
    speech_pad_ms: int = 100,
    max_speech_duration_s: float = 30.0,
    merge_gap_s: float = 0.3,
) -> list[SpeechSegment]:
    """Run Silero VAD on 16kHz mono audio and return speech segments."""
    wav_tensor = torch.from_numpy(audio).float()

    timestamps = get_speech_timestamps(
        wav_tensor,
        model,
        threshold=threshold,
        sampling_rate=SAMPLE_RATE,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        max_speech_duration_s=max_speech_duration_s,
        return_seconds=False,
    )

    segments = [
        SpeechSegment(
            start=ts["start"] / SAMPLE_RATE,
            end=ts["end"] / SAMPLE_RATE,
        )
        for ts in timestamps
    ]

    if merge_gap_s > 0:
        segments = _merge_close_segments(segments, merge_gap_s)

    return segments


def _merge_close_segments(
    segments: list[SpeechSegment], gap_s: float
) -> list[SpeechSegment]:
    if not segments:
        return segments
    merged = [segments[0]]
    for seg in segments[1:]:
        if seg.start - merged[-1].end <= gap_s:
            merged[-1] = SpeechSegment(start=merged[-1].start, end=seg.end)
        else:
            merged.append(seg)
    return merged


def extract_segment_audio(
    audio: np.ndarray, segment: SpeechSegment
) -> np.ndarray:
    """Slice audio array for a speech segment."""
    start_sample = max(0, int(segment.start * SAMPLE_RATE))
    end_sample = min(len(audio), int(segment.end * SAMPLE_RATE))
    return audio[start_sample:end_sample]
