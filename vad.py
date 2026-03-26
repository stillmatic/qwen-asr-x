from dataclasses import dataclass
from typing import Optional

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


def _otsu_threshold(probs: np.ndarray) -> float:
    """Find optimal threshold to separate speech/non-speech using Otsu's method."""
    nbins = 256
    hist, bin_edges = np.histogram(probs, bins=nbins, range=(0.0, 1.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total = hist.sum()
    if total == 0:
        return 0.5

    cum_sum = np.cumsum(hist)
    cum_mean = np.cumsum(hist * bin_centers)
    global_mean = cum_mean[-1]

    best_thresh = 0.5
    best_variance = -1.0

    for i in range(nbins - 1):
        w0 = cum_sum[i]
        w1 = total - w0
        if w0 == 0 or w1 == 0:
            continue
        mean0 = cum_mean[i] / w0
        mean1 = (global_mean - cum_mean[i]) / w1
        variance = w0 * w1 * (mean0 - mean1) ** 2
        if variance > best_variance:
            best_variance = variance
            best_thresh = bin_centers[i]

    return float(best_thresh)


@torch.no_grad()
def get_speech_probs(
    audio: np.ndarray,
    model,
    window_size_samples: int = 512,
) -> list[float]:
    """Run Silero VAD and return per-window speech probabilities."""
    wav_tensor = torch.from_numpy(audio).float()
    model.reset_states()
    probs = []
    for i in range(0, len(wav_tensor), window_size_samples):
        chunk = wav_tensor[i : i + window_size_samples]
        if len(chunk) < window_size_samples:
            chunk = torch.nn.functional.pad(
                chunk, (0, window_size_samples - len(chunk))
            )
        probs.append(model(chunk, SAMPLE_RATE).item())
    return probs


def _segment_from_probs(
    probs: list[float],
    *,
    threshold: float,
    window_size_samples: int,
    min_speech_duration_ms: int,
    min_silence_duration_ms: int,
    speech_pad_ms: int,
    max_speech_duration_s: float,
) -> list[SpeechSegment]:
    """Apply threshold to pre-computed probabilities and extract speech segments.

    Mirrors silero's get_speech_timestamps logic but skips the model inference step.
    """
    neg_threshold = threshold - 0.15
    window_sec = window_size_samples / SAMPLE_RATE
    min_speech_windows = max(1, int(min_speech_duration_ms / 1000 / window_sec))
    min_silence_windows = max(1, int(min_silence_duration_ms / 1000 / window_sec))
    max_speech_windows = int(max_speech_duration_s / window_sec) if max_speech_duration_s < float("inf") else len(probs)
    pad_samples = int(speech_pad_ms / 1000 * SAMPLE_RATE)

    # Find raw speech regions
    raw: list[tuple[int, int]] = []  # (start_window, end_window)
    in_speech = False
    start = 0
    silence_count = 0
    speech_count = 0

    for i, p in enumerate(probs):
        if not in_speech:
            if p >= threshold:
                in_speech = True
                start = i
                silence_count = 0
                speech_count = 1
        else:
            speech_count += 1
            if p < neg_threshold:
                silence_count += 1
                if silence_count >= min_silence_windows:
                    end = i - silence_count + 1
                    if end - start >= min_speech_windows:
                        raw.append((start, end))
                    in_speech = False
            else:
                silence_count = 0

            # Force-split at max duration
            if speech_count >= max_speech_windows:
                raw.append((start, i + 1))
                in_speech = False
                speech_count = 0

    if in_speech:
        end = len(probs)
        if end - start >= min_speech_windows:
            raw.append((start, end))

    # Convert window indices to sample-based SpeechSegments with padding
    total_samples = len(probs) * window_size_samples
    segments = []
    for ws, we in raw:
        s_sample = max(0, ws * window_size_samples - pad_samples)
        e_sample = min(total_samples, we * window_size_samples + pad_samples)
        segments.append(SpeechSegment(
            start=s_sample / SAMPLE_RATE,
            end=e_sample / SAMPLE_RATE,
        ))

    return segments


def detect_speech(
    audio: np.ndarray,
    model,
    *,
    threshold: Optional[float] = None,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 200,
    speech_pad_ms: int = 100,
    max_speech_duration_s: float = 30.0,
    merge_gap_s: float = 0.3,
    window_size_samples: int = 512,
) -> list[SpeechSegment]:
    """Run Silero VAD on 16kHz mono audio and return speech segments.

    threshold: Speech probability threshold. None means auto (Otsu's method).
    window_size_samples: 512 (32ms, most precise) or 1536 (96ms, 3x faster).
    """
    if threshold is None:
        # Auto mode: collect probs, pick threshold via Otsu, segment from probs
        import sys
        probs = get_speech_probs(audio, model, window_size_samples)
        threshold = _otsu_threshold(np.array(probs))
        # Clamp to reasonable range
        threshold = max(0.05, min(0.95, threshold))
        print(f"[VAD] auto threshold (Otsu): {threshold:.3f}", file=sys.stderr)

        segments = _segment_from_probs(
            probs,
            threshold=threshold,
            window_size_samples=window_size_samples,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            max_speech_duration_s=max_speech_duration_s,
        )
    else:
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
            window_size_samples=window_size_samples,
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
