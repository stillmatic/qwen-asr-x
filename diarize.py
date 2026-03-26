from collections import deque
from dataclasses import dataclass
from typing import Iterable

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
    if not segments or not turns:
        return segments

    ordered_turns = sorted(turns, key=lambda turn: (turn.start, turn.end))
    targets: list[tuple[float, float, TranscriptSegment | WordSegment]] = []
    for seg in segments:
        targets.append((seg.start, seg.end, seg))
        for word in seg.words:
            targets.append((word.start, word.end, word))

    targets.sort(key=lambda item: (item[0], item[1]))

    next_turn_idx = 0
    active_turns: deque[SpeakerTurn] = deque()
    for start, end, target in targets:
        while next_turn_idx < len(ordered_turns) and ordered_turns[next_turn_idx].start < end:
            active_turns.append(ordered_turns[next_turn_idx])
            next_turn_idx += 1

        while active_turns and active_turns[0].end <= start:
            active_turns.popleft()

        target.speaker = _find_best_overlap_speaker(start, end, active_turns)

    return segments


def _find_best_overlap_speaker(start: float, end: float, turns: Iterable[SpeakerTurn]) -> str | None:
    """Find the speaker with the most overlap for a given time range."""
    best_speaker = None
    best_overlap = 0.0
    for turn in turns:
        overlap = min(end, turn.end) - max(start, turn.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn.speaker
    return best_speaker
