import argparse
import os
import sys

from output import write_output
from pipeline import PipelineConfig, run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="WhisperX-style ASR pipeline using Qwen3-ASR + Silero VAD"
    )
    parser.add_argument("audio", help="Input audio file (any ffmpeg-supported format)")
    parser.add_argument(
        "-o", "--output", required=True, help="Output file (.json or .srt)"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-ASR-1.7B",
        help="ASR model (default: Qwen/Qwen3-ASR-1.7B)",
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
        "--device", default="cuda:0", help="Device (default: cuda:0)"
    )
    parser.add_argument(
        "--language", default=None, help="Force language (auto-detect if omitted)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Inference batch size (default: 4)",
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
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    config = PipelineConfig(
        model=args.model,
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
    )

    segments = run_pipeline(args.audio, config)
    write_output(segments, args.output)
    print(f"Output written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
