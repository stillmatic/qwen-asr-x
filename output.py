import json
from dataclasses import asdict, dataclass, field


@dataclass
class WordSegment:
    word: str
    start: float  # seconds
    end: float  # seconds
    speaker: str | None = None


@dataclass
class TranscriptSegment:
    start: float  # seconds
    end: float  # seconds
    text: str
    language: str = ""
    speaker: str | None = None
    words: list[WordSegment] = field(default_factory=list)


def to_json(segments: list[TranscriptSegment]) -> str:
    return json.dumps([asdict(s) for s in segments], indent=2, ensure_ascii=False)


def to_srt(segments: list[TranscriptSegment]) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        start = _format_srt_time(seg.start)
        end = _format_srt_time(seg.end)
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        text = seg.text
        if seg.speaker:
            text = f"[{seg.speaker}] {text}"
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def _format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_output(segments: list[TranscriptSegment], path: str) -> None:
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    if ext == "srt":
        content = to_srt(segments)
    elif ext == "json":
        content = to_json(segments)
    else:
        content = to_json(segments)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
