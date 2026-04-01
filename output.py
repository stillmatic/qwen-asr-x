import json
import re
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


_SENTENCE_END = re.compile(r'[.!?]$')


def split_segments(
    segments: list[TranscriptSegment], max_sentences: int = 2
) -> list[TranscriptSegment]:
    """Split long transcript segments into shorter ones (max N sentences each).

    Uses word-level timestamps when available, otherwise interpolates
    proportionally by character count.
    """
    result: list[TranscriptSegment] = []
    for seg in segments:
        if seg.words:
            _split_with_words(seg, max_sentences, result)
        else:
            _split_by_text(seg, max_sentences, result)
    return result


def _split_with_words(
    seg: TranscriptSegment, max_sentences: int, out: list[TranscriptSegment]
) -> None:
    sub_words: list[WordSegment] = []
    sentence_count = 0
    for word in seg.words:
        sub_words.append(word)
        if _SENTENCE_END.search(word.word.rstrip()):
            sentence_count += 1
            if sentence_count >= max_sentences:
                out.append(_sub_segment(seg, sub_words))
                sub_words = []
                sentence_count = 0
    if sub_words:
        out.append(_sub_segment(seg, sub_words))


def _sub_segment(
    parent: TranscriptSegment, words: list[WordSegment]
) -> TranscriptSegment:
    return TranscriptSegment(
        start=words[0].start,
        end=words[-1].end,
        text=" ".join(w.word for w in words),
        language=parent.language,
        speaker=parent.speaker,
        words=list(words),
    )


def _split_by_text(
    seg: TranscriptSegment, max_sentences: int, out: list[TranscriptSegment]
) -> None:
    sentences = re.split(r'(?<=[.!?])\s+', seg.text)
    if len(sentences) <= max_sentences:
        out.append(seg)
        return
    total_dur = seg.end - seg.start
    total_chars = max(len(seg.text), 1)
    char_pos = 0
    for i in range(0, len(sentences), max_sentences):
        chunk = sentences[i : i + max_sentences]
        chunk_text = " ".join(chunk)
        start_frac = char_pos / total_chars
        char_pos += len(chunk_text)
        # account for the space between chunks
        if i + max_sentences < len(sentences):
            char_pos += 1
        end_frac = min(char_pos / total_chars, 1.0)
        out.append(
            TranscriptSegment(
                start=round(seg.start + total_dur * start_frac, 3),
                end=round(seg.start + total_dur * end_frac, 3),
                text=chunk_text,
                language=seg.language,
                speaker=seg.speaker,
            )
        )


def to_json(segments: list[TranscriptSegment]) -> str:
    return json.dumps([asdict(s) for s in segments], indent=2, ensure_ascii=False)


def to_srt(segments: list[TranscriptSegment], max_sentences: int = 2) -> str:
    segments = split_segments(segments, max_sentences)
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
