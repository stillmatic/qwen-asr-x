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
    ) -> list[ASRResult]: ...


class QwenBackend:
    def __init__(
        self,
        model: str,
        device: str,
        batch_size: int,
        gpu_memory_utilization: float,
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
            max_model_len=2048,
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
    ) -> list[ASRResult]:
        transcriptions = self.asr.transcribe(
            audio=audio,
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
    ) -> list[ASRResult]:
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


def create_backend(config) -> ASRBackend:
    """Create an ASR backend from a PipelineConfig."""
    if config.backend == "cohere":
        return CohereBackend(
            model=config.model,
            device=config.device,
            batch_size=config.batch_size,
            hf_token=config.hf_token,
        )
    # default: qwen
    return QwenBackend(
        model=config.model,
        device=config.device,
        batch_size=config.batch_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        enforce_eager=config.enforce_eager,
        hf_token=config.hf_token,
    )
