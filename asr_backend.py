from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np


@dataclass
class ASRResult:
    text: str
    language: str = ""


class ASRBackend(Protocol):
    def transcribe(
        self,
        audio: list[tuple[np.ndarray, int]],
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> list[ASRResult]: ...


class QwenBackend:
    def __init__(
        self,
        model: str,
        device: str,
        batch_size: int,
        gpu_memory_utilization: float,
        max_model_len: int,
        enforce_eager: bool,
        hf_token: Optional[str],
    ):
        import torch
        from qwen_asr import Qwen3ASRModel

        kwargs = dict(
            model=model,
            forced_aligner=None,
            max_inference_batch_size=batch_size,
            max_new_tokens=512,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype="bfloat16",
            hf_token=hf_token,
        )
        if enforce_eager:
            kwargs["enforce_eager"] = True
        self.asr = Qwen3ASRModel.LLM(**kwargs)

    def transcribe(
        self,
        audio: list[tuple[np.ndarray, int]],
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> list[ASRResult]:
        transcriptions = self.asr.transcribe(
            audio=audio,
            context=prompt or "",
            language=language,
            return_time_stamps=False,
        )
        return [
            ASRResult(text=(tx.text or "").strip(), language=tx.language or "")
            for tx in transcriptions
        ]


class CohereBackend:
    def __init__(
        self,
        model: str,
        device: str,
        batch_size: int,
        hf_token: Optional[str],
    ):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            model, trust_remote_code=True, token=hf_token,
        )
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model, trust_remote_code=True, token=hf_token,
        ).to(device)
        self.model.eval()
        self.batch_size = batch_size

    def transcribe(
        self,
        audio: list[tuple[np.ndarray, int]],
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> list[ASRResult]:
        if not language:
            language = "en"
        audio_arrays = [arr for arr, _ in audio]
        sample_rates = [sr for _, sr in audio]
        texts = self.model.transcribe(
            processor=self.processor,
            audio_arrays=audio_arrays,
            sample_rates=sample_rates,
            language=language,
            batch_size=self.batch_size,
            compile=True,
        )
        return [
            ASRResult(text=(t or "").strip(), language=language or "")
            for t in texts
        ]


class WhisperBackend:
    """Whisper via vLLM encoder-decoder inference."""

    # Map common language names to Whisper language tokens
    _LANG_MAP = {
        "english": "en", "chinese": "zh", "german": "de", "spanish": "es",
        "russian": "ru", "korean": "ko", "french": "fr", "japanese": "ja",
        "portuguese": "pt", "turkish": "tr", "polish": "pl", "catalan": "ca",
        "dutch": "nl", "arabic": "ar", "swedish": "sv", "italian": "it",
        "indonesian": "id", "hindi": "hi", "finnish": "fi", "vietnamese": "vi",
        "hebrew": "he", "ukrainian": "uk", "greek": "el", "malay": "ms",
        "czech": "cs", "romanian": "ro", "danish": "da", "hungarian": "hu",
        "tamil": "ta", "norwegian": "no", "thai": "th", "urdu": "ur",
        "croatian": "hr", "bulgarian": "bg", "lithuanian": "lt", "latin": "la",
        "maori": "mi", "malayalam": "ml", "welsh": "cy", "slovak": "sk",
        "telugu": "te", "persian": "fa", "latvian": "lv", "bengali": "bn",
        "serbian": "sr", "azerbaijani": "az", "slovenian": "sl", "kannada": "kn",
        "estonian": "et", "macedonian": "mk", "breton": "br", "basque": "eu",
        "icelandic": "is", "armenian": "hy", "nepali": "ne", "mongolian": "mn",
        "bosnian": "bs", "kazakh": "kk", "albanian": "sq", "swahili": "sw",
        "galician": "gl", "marathi": "mr", "punjabi": "pa", "sinhala": "si",
        "khmer": "km", "shona": "sn", "yoruba": "yo", "somali": "so",
        "afrikaans": "af", "occitan": "oc", "georgian": "ka", "belarusian": "be",
        "tajik": "tg", "sindhi": "sd", "gujarati": "gu", "amharic": "am",
        "yiddish": "yi", "lao": "lo", "uzbek": "uz", "faroese": "fo",
        "haitian creole": "ht", "pashto": "ps", "turkmen": "tk",
        "nynorsk": "nn", "maltese": "mt", "sanskrit": "sa", "luxembourgish": "lb",
        "myanmar": "my", "tibetan": "bo", "tagalog": "tl", "malagasy": "mg",
        "assamese": "as", "tatar": "tt", "hawaiian": "haw", "lingala": "ln",
        "hausa": "ha", "bashkir": "ba", "javanese": "jw", "sundanese": "su",
    }

    def __init__(
        self,
        model: str,
        batch_size: int,
        gpu_memory_utilization: float,
        enforce_eager: bool,
        hf_token: Optional[str],
    ):
        from vllm import LLM

        kwargs = dict(
            model=model,
            max_model_len=448,
            max_num_seqs=batch_size,
            limit_mm_per_prompt={"audio": 1},
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="float16",
        )
        if enforce_eager:
            kwargs["enforce_eager"] = True
        if hf_token:
            kwargs["hf_token"] = hf_token
        self.llm = LLM(**kwargs)
        self.batch_size = batch_size

    def _build_prompt(self, language: Optional[str], prompt_text: Optional[str] = None) -> str:
        prompt = "<|startoftranscript|>"
        if language:
            lang_code = self._LANG_MAP.get(language.lower(), language.lower())
            prompt += f"<|{lang_code}|>"
        prompt += "<|transcribe|><|notimestamps|>"
        if prompt_text:
            prompt += f" {prompt_text}"
        return prompt

    def transcribe(
        self,
        audio: list[tuple[np.ndarray, int]],
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> list[ASRResult]:
        from vllm import SamplingParams

        decoder_prompt = self._build_prompt(language, prompt)
        sampling_params = SamplingParams(
            temperature=0,
            top_p=1.0,
            max_tokens=448,
        )

        prompts = [
            {
                "prompt": decoder_prompt,
                "multi_modal_data": {
                    "audio": (arr, sr),
                },
            }
            for arr, sr in audio
        ]

        outputs = self.llm.generate(prompts, sampling_params)

        lang_str = language or ""
        return [
            ASRResult(
                text=output.outputs[0].text.strip(),
                language=lang_str,
            )
            for output in outputs
        ]


def create_backend(config) -> ASRBackend:
    """Create an ASR backend from a PipelineConfig."""
    if config.backend == "cohere":
        return CohereBackend(
            model=config.model,
            device=config.device,
            batch_size=config.batch_size,
            hf_token=config.hf_token,
        )
    if config.backend == "whisper":
        return WhisperBackend(
            model=config.model,
            batch_size=config.batch_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enforce_eager=config.enforce_eager,
            hf_token=config.hf_token,
        )
    # default: qwen
    return QwenBackend(
        model=config.model,
        device=config.device,
        batch_size=config.batch_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_model_len,
        enforce_eager=config.enforce_eager,
        hf_token=config.hf_token,
    )
