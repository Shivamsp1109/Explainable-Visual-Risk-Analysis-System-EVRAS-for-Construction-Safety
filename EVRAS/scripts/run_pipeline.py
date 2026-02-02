import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from pipeline import analyze_image


def main():
    parser = argparse.ArgumentParser(description="Run EVRAS inference pipeline.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--output-image", required=True, help="Path to save annotated image.")
    parser.add_argument("--output-json", required=True, help="Path to save JSON output.")
    parser.add_argument("--include-llm", default="true", help="true|false to include LLM explanation.")

    args = parser.parse_args()
    include_llm = str(args.include_llm).lower() not in ["false", "0", "no"]

    output, _ = analyze_image(
        image_path=args.image,
        include_llm=include_llm,
        output_json_path=args.output_json,
        output_image_path=args.output_image
    )

    print(json.dumps(output))


if __name__ == "__main__":
    main()
