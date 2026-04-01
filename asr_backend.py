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


class FireRedAEDBackend:
    """FireRedASR2-AED via native fireredasr2s library."""

    def __init__(
        self,
        model: str,
        device: str,
        batch_size: int,
    ):
        from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config

        asr_config = FireRedAsr2Config(
            use_gpu="cuda" in device,
            use_half=False,
            beam_size=3,
            nbest=1,
            decode_max_len=0,
            softmax_smoothing=1.25,
            aed_length_penalty=0.6,
            eos_penalty=1.0,
            return_timestamp=False,
        )
        self.model = FireRedAsr2.from_pretrained("aed", model, asr_config)
        self.batch_size = batch_size

    def transcribe(
        self,
        audio: list[tuple[np.ndarray, int]],
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> list[ASRResult]:
        import os
        import tempfile
        import wave

        results: list[ASRResult] = []
        for i in range(0, len(audio), self.batch_size):
            batch = audio[i : i + self.batch_size]
            tmp_paths: list[str] = []
            uttids: list[str] = []
            try:
                for j, (arr, sr) in enumerate(batch):
                    fd, path = tempfile.mkstemp(suffix=".wav")
                    os.close(fd)
                    tmp_paths.append(path)
                    uttids.append(f"seg_{i + j}")
                    int16 = (arr * 32767).clip(-32768, 32767).astype(np.int16)
                    with wave.open(path, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(sr)
                        wf.writeframes(int16.tobytes())

                batch_results = self.model.transcribe(uttids, tmp_paths)
                for r in batch_results:
                    results.append(
                        ASRResult(
                            text=(r.get("text", "") or "").strip(),
                            language="",
                        )
                    )
            finally:
                for p in tmp_paths:
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

        return results


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

    # vLLM handles its own memory/scheduling, so we pass all segments in
    # large batches and let the engine figure out parallelism.
    preferred_batch_size = 256

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
            max_num_seqs=self.preferred_batch_size,
            limit_mm_per_prompt={"audio": 1},
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="float16",
        )
        if enforce_eager:
            kwargs["enforce_eager"] = True
        if hf_token:
            kwargs["hf_token"] = hf_token
        self.llm = LLM(**kwargs)

    def _build_prompt(self, language: Optional[str]) -> str:
        prompt = "<|startoftranscript|>"
        if language:
            lang_code = self._LANG_MAP.get(language.lower(), language.lower())
            prompt += f"<|{lang_code}|>"
        return prompt

    def transcribe(
        self,
        audio: list[tuple[np.ndarray, int]],
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> list[ASRResult]:
        from vllm import SamplingParams

        decoder_prompt = self._build_prompt(language)
        sampling_params = SamplingParams(
            temperature=0,
            top_p=1.0,
            max_tokens=200,
        )

        prompts = [
            {
                "encoder_prompt": {
                    "prompt": "",
                    "multi_modal_data": {
                        "audio": (arr, sr),
                    },
                },
                "decoder_prompt": decoder_prompt,
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
    if config.backend == "firered":
        return FireRedAEDBackend(
            model=config.model,
            device=config.device,
            batch_size=config.batch_size,
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
