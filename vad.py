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


@dataclass
class VadResult:
    segments: list[SpeechSegment]
    params: dict  # the actual params used (threshold, min_speech_duration_ms, etc.)


def load_vad_model(backend: str = "silero"):
    if backend == "firered":
        return _load_firered_vad()
    return load_silero_vad()


def _load_firered_vad():
    from huggingface_hub import snapshot_download
    from fireredvad import FireRedVad, FireRedVadConfig

    model_dir = snapshot_download("FireRedTeam/FireRedVAD")
    vad_dir = f"{model_dir}/VAD"
    config = FireRedVadConfig(
        use_gpu=torch.cuda.is_available(),
        smooth_window_size=5,
        speech_threshold=0.4,
        min_speech_frame=20,
        max_speech_frame=3000,
        min_silence_frame=10,
        merge_silence_frame=0,
        extend_speech_frame=0,
        chunk_max_frame=30000,
    )
    return FireRedVad.from_pretrained(vad_dir, config)


@dataclass
class AutoVadParams:
    threshold: float
    min_speech_duration_ms: int
    min_silence_duration_ms: int
    merge_gap_s: float
    speech_pad_ms: int = 80


def _otsu(values: np.ndarray, nbins: int = 256, lo: float = 0.0, hi: float = 1.0) -> float:
    """Otsu's method: find threshold that best separates a bimodal distribution."""
    hist, bin_edges = np.histogram(values, bins=nbins, range=(lo, hi))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total = hist.sum()
    if total == 0:
        return (lo + hi) / 2

    cum_sum = np.cumsum(hist)
    cum_mean = np.cumsum(hist * bin_centers)
    global_mean = cum_mean[-1]

    best_thresh = (lo + hi) / 2
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


def _auto_vad_params(
    probs: list[float],
    window_size_samples: int,
) -> AutoVadParams:
    """Derive all VAD params from the probability curve."""
    prob_arr = np.array(probs)
    window_ms = window_size_samples / SAMPLE_RATE * 1000

    # 1. Threshold: find where the silence cluster ends.
    #    Otsu tends to pick too high when silence dominates. Instead:
    #    - Find the silence peak (mode of values < 0.5)
    #    - Set threshold at silence_peak + 3*sigma of the silence cluster
    #    - Fall back to Otsu if the distribution is unusual
    silence_probs = prob_arr[prob_arr < 0.5]
    if len(silence_probs) > 10:
        silence_mean = float(np.mean(silence_probs))
        silence_std = float(np.std(silence_probs))
        # Threshold just above the silence noise floor
        threshold = silence_mean + max(3 * silence_std, 0.05)
        threshold = max(0.05, min(0.5, threshold))
    else:
        # Very little silence — fall back to Otsu
        threshold = _otsu(prob_arr)
        threshold = max(0.05, min(0.95, threshold))

    # 2. Find raw above-threshold regions (no duration filtering yet)
    above = prob_arr >= threshold
    regions: list[tuple[int, int]] = []  # (start_idx, end_idx)
    in_region = False
    start = 0
    for i, v in enumerate(above):
        if v and not in_region:
            in_region = True
            start = i
        elif not v and in_region:
            regions.append((start, i))
            in_region = False
    if in_region:
        regions.append((start, len(above)))

    # 3. Compute gap durations and speech durations in ms
    speech_durations = np.array([(e - s) * window_ms for s, e in regions])
    gaps = np.array([
        (regions[i + 1][0] - regions[i][1]) * window_ms
        for i in range(len(regions) - 1)
    ]) if len(regions) > 1 else np.array([])

    # 4. min_silence_duration: Otsu on gap durations
    #    Separates intra-utterance pauses from real silence breaks
    if len(gaps) >= 3:
        gap_split = _otsu(gaps, nbins=64, lo=float(gaps.min()), hi=float(gaps.max()))
        min_silence_ms = int(max(50, min(gap_split, 1000)))
    else:
        min_silence_ms = 100

    # 5. min_speech_duration: filter noise bursts
    #    Use 10th percentile of speech durations as noise floor
    if len(speech_durations) >= 5:
        min_speech_ms = int(max(30, np.percentile(speech_durations, 10)))
    else:
        min_speech_ms = 50

    # 6. merge_gap: slightly above min_silence to bridge stutters
    merge_gap_s = round(min_silence_ms / 1000 * 1.5, 2)
    merge_gap_s = max(0.1, min(merge_gap_s, 2.0))

    return AutoVadParams(
        threshold=threshold,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
        merge_gap_s=merge_gap_s,
    )


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


def _detect_speech_firered(audio: np.ndarray, model, merge_gap_s: float) -> VadResult:
    """Run FireRedVAD on 16kHz mono float32 audio."""
    # FireRedVAD expects int16
    audio_int16 = (audio * 32767).astype(np.int16)
    result, _probs = model.detect(audio_int16)
    segments = [
        SpeechSegment(start=round(s, 3), end=round(e, 3))
        for s, e in result["timestamps"]
    ]
    if merge_gap_s > 0:
        segments = _merge_close_segments(segments, merge_gap_s)
    return VadResult(segments=segments, params={"backend": "firered"})


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
) -> VadResult:
    """Run VAD on 16kHz mono audio and return speech segments.

    Supports Silero VAD models and FireRedVAD models.
    threshold: Speech probability threshold. None means auto (Otsu's method),
               which also auto-derives min_speech, min_silence, pad, and merge_gap.
    """
    # FireRedVAD path
    try:
        from fireredvad import FireRedVad
        if isinstance(model, FireRedVad):
            return _detect_speech_firered(audio, model, merge_gap_s)
    except ImportError:
        pass
    if threshold is None:
        # Auto mode: derive all params from the probability curve
        import sys
        probs = get_speech_probs(audio, model, window_size_samples)
        auto = _auto_vad_params(probs, window_size_samples)
        threshold = auto.threshold
        min_speech_duration_ms = auto.min_speech_duration_ms
        min_silence_duration_ms = auto.min_silence_duration_ms
        speech_pad_ms = auto.speech_pad_ms
        merge_gap_s = auto.merge_gap_s
        print(
            f"[VAD] auto params — threshold: {threshold:.3f}, "
            f"min_speech: {min_speech_duration_ms}ms, "
            f"min_silence: {min_silence_duration_ms}ms, "
            f"pad: {speech_pad_ms}ms, "
            f"merge_gap: {merge_gap_s}s",
            file=sys.stderr,
        )

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

    return VadResult(
        segments=segments,
        params={
            "threshold": round(threshold, 4),
            "min_speech_duration_ms": min_speech_duration_ms,
            "min_silence_duration_ms": min_silence_duration_ms,
            "speech_pad_ms": speech_pad_ms,
            "max_speech_duration_s": max_speech_duration_s,
            "merge_gap_s": merge_gap_s,
        },
    )


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
