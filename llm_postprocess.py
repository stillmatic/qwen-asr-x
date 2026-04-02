"""LLM postprocessing for transcript correction or translation via OpenRouter API."""

import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from output import TranscriptSegment

DEBUG_DIR = Path("debug")

_DEBUG_MODELS = [
    "x-ai/grok-4.1-fast",
    "qwen/qwen3.6-plus:free",
]

_SYSTEM_FIX = (
    "You are a transcript editor. You will receive a speech-to-text transcript "
    "split into numbered segments. "
    "The transcript likely has mistakes. For context, it is a transcript of an adult video, "
    "and you need to write exactly what was likely to be said in context. "
    "For example, 'cum' might be mistranscribed as 'comb' or 'cock' as 'car'. "
    "嗯 or similar single characters should be [moan]. "
    "Fix any transcription errors (misheard words, grammar, punctuation) "
    "while keeping the meaning intact.\n\n"
    "Output ONLY the segments that need correction, using this exact format "
    "for each change:\n"
    "<segment_number>\n"
    "BEFORE: <original text>\n"
    "AFTER: <corrected text>\n\n"
    "If no segments need changes, output exactly: NO CHANGES\n"
    "Do not output segments that are already correct."
)

_SYSTEM_TRANSLATE = (
    "You are a translator. You will receive a speech-to-text transcript "
    "split into numbered segments. "
    "The transcript likely has mistakes. For context, it is a transcript of an adult video,"
    "and you need to write exactly what was likely to be said in context."
    "For example, 'cum' might be mistranscribed as 'comb' or 'car' as cock'."
    "Translate the text to English."
    "Output ONLY the translated segments in the exact same numbered format, one per line. "
    "Do not add, remove, or merge segments."
)


def _format_segments(segments: list[TranscriptSegment]) -> str:
    return "\n".join(f"{i + 1}: {seg.text}" for i, seg in enumerate(segments))


def _parse_output(text: str, n_expected: int) -> list[str] | None:
    """Parse numbered segment lines from LLM output. Returns None on failure."""
    lines = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\d+\s*[:\.]\s*(.+)$", line)
        if m:
            lines.append(m.group(1).strip())
    if len(lines) == n_expected:
        return lines
    return None


def _parse_diff_output(
    text: str, segments: list[TranscriptSegment]
) -> dict[int, str] | None:
    """Parse diff-format output. Returns {segment_index: new_text} or None on failure.

    Expected format per change:
        <number>
        BEFORE: <original>
        AFTER: <corrected>
    """
    text = text.strip()
    if text.upper() == "NO CHANGES":
        return {}

    changes: dict[int, str] = {}
    pattern = r"(\d+)\s*\n\s*BEFORE:\s*(.+)\s*\n\s*AFTER:\s*(.+)"
    for m in re.finditer(pattern, text):
        seg_num = int(m.group(1))
        before = m.group(2).strip()
        after = m.group(3).strip()
        idx = seg_num - 1  # 0-indexed
        if not (0 <= idx < len(segments)):
            _log(f"  diff: segment {seg_num} out of range (have {len(segments)})")
            continue
        orig = segments[idx].text.strip()
        if orig != before:
            _log(f"  diff: BEFORE mismatch at {seg_num}: expected '{orig}', got '{before}'")
        changes[idx] = after

    if not changes:
        return None  # had content but nothing parseable — treat as failure
    return changes


def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr)


class LLMPostprocessor:
    """Calls OpenRouter API for transcript postprocessing (fix or translate)."""

    def __init__(self, model: str, debug: bool = False):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY environment variable is required for LLM postprocessing"
            )
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = model
        self.debug = debug
        _log(f"LLM postprocessor ready (OpenRouter model: {model}, debug={debug})")

    def _call_model(self, model: str, messages: list[dict]) -> tuple[str, str]:
        """Call a single model, return (raw_content, stripped_content)."""
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=16384,
            timeout=300,
        )
        raw = response.choices[0].message.content or ""
        # Strip thinking for parsed output but keep raw for debug
        stripped = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL).strip()
        return raw, stripped

    def process(
        self,
        segments: list[TranscriptSegment],
        mode: str = "fix",
    ) -> list[TranscriptSegment]:
        """Postprocess transcript segments. mode: 'fix' or 'translate'."""
        if not segments:
            return segments

        system = _SYSTEM_TRANSLATE if mode == "translate" else _SYSTEM_FIX
        user_text = _format_segments(segments)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ]

        n_expected = len(segments)

        if self.debug:
            return self._process_debug(messages, segments, mode, n_expected)

        _log(f"LLM postprocess ({mode}): {n_expected} segments via OpenRouter ({self.model})")

        raw, stripped = self._call_model(self.model, messages)
        _log(f"LLM raw output ({len(stripped)} chars):\n{stripped}")

        return self._apply_output(stripped, segments, mode, n_expected, messages)

    def _process_debug(
        self,
        messages: list[dict],
        segments: list[TranscriptSegment],
        mode: str,
        n_expected: int,
    ) -> list[TranscriptSegment]:
        """Call all debug models concurrently, stream-print as they finish, save to debug/."""
        models = [self.model] + [m for m in _DEBUG_MODELS if m != self.model]
        _log(
            f"LLM DEBUG ({mode}): {n_expected} segments, "
            f"calling {len(models)} models in parallel"
        )

        sep = "=" * 70
        print(f"\n{sep}", file=sys.stderr)
        print(f"LLM DEBUG COMPARISON ({mode})", file=sys.stderr)
        print(f"Input segments:", file=sys.stderr)
        print(f"{_format_segments(segments)}", file=sys.stderr)
        print(sep, file=sys.stderr, flush=True)

        results: dict[str, tuple[str, str] | str] = {}  # model -> (raw, stripped) or error str

        def call(model: str):
            return model, self._call_model(model, messages)

        primary_stripped = None

        with ThreadPoolExecutor(max_workers=len(models)) as pool:
            futures = {pool.submit(call, m): m for m in models}
            for future in as_completed(futures):
                model = futures[future]
                try:
                    _, (raw, stripped) = future.result()
                    results[model] = (raw, stripped)
                    if model == self.model:
                        primary_stripped = stripped
                    if mode == "fix":
                        diff_parsed = _parse_diff_output(stripped, segments)
                        status = f"DIFF OK: {len(diff_parsed)} changes" if diff_parsed is not None else "DIFF PARSE FAILED"
                    else:
                        parsed = _parse_output(stripped, n_expected)
                        status = f"PARSE OK: {len(parsed)} segments" if parsed else f"PARSE FAILED: expected {n_expected}"
                    print(f"\n--- {model} [{status}] ---", file=sys.stderr)
                    print(raw, file=sys.stderr, flush=True)
                except Exception as e:
                    results[model] = str(e)
                    print(f"\n--- {model} [ERROR] ---", file=sys.stderr)
                    print(f"  {e}", file=sys.stderr, flush=True)

        print(f"\n{sep}\n", file=sys.stderr, flush=True)

        # Save all results to debug/
        DEBUG_DIR.mkdir(exist_ok=True)
        ts = int(time.time())
        debug_file = DEBUG_DIR / f"llm_debug_{mode}_{ts}.txt"
        parts = [
            f"=== INPUT ({n_expected} segments) ===",
            _format_segments(segments),
            "",
        ]
        for model in models:
            parts.append(f"=== {model} ===")
            result = results.get(model)
            if isinstance(result, str):
                parts.append(f"ERROR: {result}")
            else:
                raw, stripped = result
                if mode == "fix":
                    diff_parsed = _parse_diff_output(stripped, segments)
                    parts.append(f"[{'DIFF OK: ' + str(len(diff_parsed)) + ' changes' if diff_parsed is not None else 'DIFF PARSE FAILED'}]")
                else:
                    parsed = _parse_output(stripped, n_expected)
                    parts.append(f"[{'PARSE OK' if parsed else 'PARSE FAILED'}]")
                parts.append(raw)
            parts.append("")
        debug_file.write_text("\n".join(parts), encoding="utf-8")
        _log(f"LLM debug saved to {debug_file}")

        # Use primary model's output for actual result
        if primary_stripped is None:
            _log(f"LLM DEBUG: primary model {self.model} failed, returning original")
            return segments

        return self._apply_output(primary_stripped, segments, mode, n_expected, messages)

    def _apply_output(
        self,
        output_text: str,
        segments: list[TranscriptSegment],
        mode: str,
        n_expected: int,
        messages: list[dict],
    ) -> list[TranscriptSegment]:
        """Parse LLM output and apply to segments."""
        if mode == "fix":
            return self._apply_diff_output(output_text, segments, messages)

        # translate mode: full numbered output
        parsed = _parse_output(output_text, n_expected)
        if parsed is None:
            self._save_debug(mode, output_text, n_expected, messages)
            return segments

        original_texts = [seg.text for seg in segments]
        for seg, new_text in zip(segments, parsed):
            seg.text = new_text
            seg.words = []

        _log("LLM postprocess before/after:")
        for i, (orig, new_text) in enumerate(zip(original_texts, parsed)):
            if orig != new_text:
                _log(f"  {i+1} BEFORE: {orig}")
                _log(f"  {i+1} AFTER:  {new_text}")

        _log("LLM postprocess complete")
        return segments

    def _apply_diff_output(
        self,
        output_text: str,
        segments: list[TranscriptSegment],
        messages: list[dict],
    ) -> list[TranscriptSegment]:
        """Parse diff-format output and apply changes to segments."""
        changes = _parse_diff_output(output_text, segments)
        if changes is None:
            self._save_debug("fix", output_text, len(segments), messages)
            return segments

        if not changes:
            _log("LLM postprocess: no changes needed")
            return segments

        for idx, new_text in changes.items():
            _log(f"  {idx+1} BEFORE: {segments[idx].text}")
            _log(f"  {idx+1} AFTER:  {new_text}")
            segments[idx].text = new_text

        _log(f"LLM postprocess complete ({len(changes)}/{len(segments)} segments changed)")
        return segments

    def _save_debug(
        self, mode: str, output_text: str, n_expected: int, messages: list[dict]
    ):
        DEBUG_DIR.mkdir(exist_ok=True)
        debug_file = DEBUG_DIR / f"llm_{mode}_{int(time.time())}.txt"
        debug_file.write_text(
            f"=== MESSAGES ===\n{messages}\n\n"
            f"=== RAW OUTPUT ({len(output_text)} chars) ===\n{output_text}\n\n"
            f"=== EXPECTED {n_expected} segments ===\n",
            encoding="utf-8",
        )
        _log(
            f"LLM postprocess: parse failed "
            f"(got {len(output_text)} chars), debug saved to {debug_file}"
        )
