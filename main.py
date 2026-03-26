import argparse
import os
import sys

from output import write_output
from pipeline import PipelineConfig, run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="ASR pipeline with Silero VAD and pluggable backends"
    )
    parser.add_argument("audio", help="Input audio file (any ffmpeg-supported format)")
    parser.add_argument(
        "-o", "--output", required=True, help="Output file (.json or .srt)"
    )
    parser.add_argument(
        "--backend",
        choices=["qwen", "cohere"],
        default="qwen",
        help="ASR backend (default: qwen)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="ASR model (default depends on backend)",
    )
    parser.add_argument(
        "--aligner",
        default="Qwen/Qwen3-ForcedAligner-0.6B",
        help="Forced aligner model (default: Qwen/Qwen3-ForcedAligner-0.6B)",
    )
    parser.add_argument(
        "--no-align", action="store_true", help="Skip forced alignment"
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device for aligner/diarization (default: cuda:0)",
    )
    parser.add_argument(
        "--language", default=None, help="Force language (auto-detect if omitted)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Max ASR inference batch size (default: 4)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="vLLM GPU memory utilization target (qwen only, default: 0.8)",
    )
    parser.add_argument(
        "--diarize", action="store_true", help="Enable speaker diarization"
    )
    parser.add_argument(
        "--diarize-model",
        default="pyannote/speaker-diarization-community-1",
        help="Diarization model (default: pyannote/speaker-diarization-community-1)",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token for gated models (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--min-speakers", type=int, default=None, help="Min speakers for diarization"
    )
    parser.add_argument(
        "--max-speakers", type=int, default=None, help="Max speakers for diarization"
    )
    # VAD tuning
    parser.add_argument(
        "--vad-threshold", default="auto", help="VAD threshold, float or 'auto' (default: auto)"
    )
    parser.add_argument(
        "--min-speech-duration-ms", type=int, default=50,
        help="Min speech duration in ms (default: 50)",
    )
    parser.add_argument(
        "--min-silence-duration-ms", type=int, default=100,
        help="Min silence duration in ms (default: 100)",
    )
    parser.add_argument(
        "--speech-pad-ms", type=int, default=100,
        help="Speech padding in ms (default: 100)",
    )
    parser.add_argument(
        "--save-vad", default=None, help="Save VAD segments to JSON file"
    )
    parser.add_argument(
        "--visualize-vad", action="store_true",
        help="Save VAD probability plot (silero_vad_figure.png)",
    )
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    default_models = {
        "qwen": "Qwen/Qwen3-ASR-1.7B",
        "cohere": "CohereLabs/cohere-transcribe-03-2026",
    }
    model = args.model or default_models[args.backend]

    config = PipelineConfig(
        backend=args.backend,
        model=model,
        aligner=args.aligner,
        device=args.device,
        align=not args.no_align,
        diarize=args.diarize,
        diarize_model=args.diarize_model,
        hf_token=hf_token,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        language=args.language,
        batch_size=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        vad_threshold=None if args.vad_threshold == "auto" else float(args.vad_threshold),
        min_speech_duration_ms=args.min_speech_duration_ms,
        min_silence_duration_ms=args.min_silence_duration_ms,
        speech_pad_ms=args.speech_pad_ms,
        save_vad=args.save_vad,
        visualize_vad=args.visualize_vad,
    )

    segments = run_pipeline(args.audio, config)
    write_output(segments, args.output)
    print(f"Output written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
