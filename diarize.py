from dataclasses import dataclass

import numpy as np
import torch
from pyannote.audio import Pipeline as PyannotePipeline

from output import TranscriptSegment, WordSegment


SAMPLE_RATE = 16000


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: str


def load_diarization_pipeline(
    model: str = "pyannote/speaker-diarization-community-1",
    device: str = "cuda:0",
    hf_token: str | None = None,
) -> PyannotePipeline:
    pipeline = PyannotePipeline.from_pretrained(model, token=hf_token)
    pipeline.to(torch.device(device))
    return pipeline


def run_diarization(
    audio: np.ndarray,
    pipeline: PyannotePipeline,
    *,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[SpeakerTurn]:
    """Run pyannote diarization on 16kHz mono audio, return speaker turns."""
    waveform = torch.from_numpy(audio).unsqueeze(0).float()
    audio_input = {"waveform": waveform, "sample_rate": SAMPLE_RATE}

    kwargs = {}
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    result = pipeline(audio_input, **kwargs)

    # pyannote v4 returns DiarizeOutput, v3 returns Annotation directly
    annotation = getattr(result, "speaker_diarization", result)

    turns = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        turns.append(
            SpeakerTurn(
                start=round(segment.start, 3),
                end=round(segment.end, 3),
                speaker=speaker,
            )
        )
    return turns


def assign_speakers(
    segments: list[TranscriptSegment],
    turns: list[SpeakerTurn],
) -> list[TranscriptSegment]:
    """Assign speaker labels to transcript segments and their words."""
    for seg in segments:
        seg.speaker = _find_speaker(seg.start, seg.end, turns)
        for word in seg.words:
            word.speaker = _find_speaker(word.start, word.end, turns)
    return segments


def _find_speaker(start: float, end: float, turns: list[SpeakerTurn]) -> str | None:
    """Find the speaker with the most overlap for a given time range."""
    best_speaker = None
    best_overlap = 0.0
    for turn in turns:
        overlap = min(end, turn.end) - max(start, turn.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn.speaker
    return best_speaker
