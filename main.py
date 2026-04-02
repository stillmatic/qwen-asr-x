import argparse
import sys

from dotenv import load_dotenv

from output import write_output
from pipeline import add_pipeline_args, config_from_args, run_pipeline

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="ASR pipeline with Silero VAD and pluggable backends"
    )
    parser.add_argument("audio", help="Input audio file (any ffmpeg-supported format)")
    parser.add_argument(
        "-o", "--output", required=True, help="Output file (.json or .srt)"
    )
    add_pipeline_args(parser)
    # main.py-only args
    parser.add_argument(
        "--no-align", action="store_true", help="Skip forced alignment"
    )
    parser.add_argument(
        "--language", default=None, help="Force language (auto-detect if omitted)"
    )
    parser.add_argument(
        "--save-vad", default=None, help="Save VAD segments to JSON file"
    )
    args = parser.parse_args()

    config = config_from_args(args)
    segments = run_pipeline(args.audio, config)
    write_output(segments, args.output)
    print(f"Output written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
