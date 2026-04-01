"""LLM postprocessing for transcript correction or translation."""

import re
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_float32_matmul_precision("high")

from output import TranscriptSegment

DEBUG_DIR = Path("debug")



_SYSTEM_FIX = (
    "You are a transcript editor. You will receive a speech-to-text transcript "
    "split into numbered segments."
    "The transcript likely has mistakes. For context, it is a transcript of an adult video,"
    "and you need to write exactly what was likely to be said in context."
    "For example, 'cum' might be mistranscribed as 'comb' or 'cock' as 'car'."
    "嗯 or similar single characters should be [moan]"
    "Fix any transcription errors (misheard words, "
    "grammar, punctuation) while keeping the meaning and segment structure intact. "
    "Output ONLY the corrected segments in the exact same numbered format, one per line. "
    "Do not add, remove, or merge segments."
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


def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr)


class LLMPostprocessor:
    """Loads a text LLM for transcript postprocessing (fix or translate)."""

    def __init__(self, model: str, device: str = "cuda:0"):
        _log(f"Loading LLM postprocessor: {model}")
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        _log("LLM postprocessor loaded")

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

        # Enable thinking for Qwen3 (strip it from output later)
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        _log(f"LLM postprocess ({mode}): {len(segments)} segments, {input_len} input tokens")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max(input_len * 2, 1024),
                do_sample=False,
            )

        new_tokens = outputs[0][input_len:]
        # Decode with special tokens so we can strip <think>...</think> properly
        output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        # Strip thinking block
        output_text = re.sub(
            r"<think>.*?</think>\s*", "", output_text, flags=re.DOTALL
        )
        # Strip any remaining special tokens
        output_text = re.sub(r"<\|[^>]+\|>", "", output_text).strip()

        _log(f"LLM raw output ({len(output_text)} chars):\n{output_text}")

        parsed = _parse_output(output_text, len(segments))
        if parsed is None:
            # Dump debug info
            DEBUG_DIR.mkdir(exist_ok=True)
            debug_file = DEBUG_DIR / f"llm_{mode}_{int(time.time())}.txt"
            debug_file.write_text(
                f"=== PROMPT ===\n{text}\n\n"
                f"=== RAW OUTPUT ({len(output_text)} chars) ===\n{output_text}\n\n"
                f"=== EXPECTED {len(segments)} segments ===\n",
                encoding="utf-8",
            )
            _log(
                f"LLM postprocess: parse failed "
                f"(got {len(output_text)} chars), debug saved to {debug_file}"
            )
            return segments

        original_texts = [seg.text for seg in segments]
        for seg, new_text in zip(segments, parsed):
            seg.text = new_text
            if mode == "translate":
                seg.words = []  # word alignment invalid after translation

        _log("LLM postprocess before/after:")
        for i, (orig, new_text) in enumerate(zip(original_texts, parsed)):
            if orig != new_text:
                _log(f"  {i+1} BEFORE: {orig}")
                _log(f"  {i+1} AFTER:  {new_text}")

        _log("LLM postprocess complete")
        return segments
