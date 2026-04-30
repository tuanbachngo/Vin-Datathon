"""Create globally scaled versions of a submission file.

Use this for post-processing around the current best submission.

Formula:
    scaled_revenue = Revenue * scale

Example:
    python scale_submission_global.py --input "submissions/submission_feature_rich_blend_refined5.csv" --scales "0.98,0.99,1.01,1.02"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_scales(text: str) -> list[float]:
    scales = []
    for part in text.split(","):
        part = part.strip()
        if part:
            scales.append(float(part))
    if not scales:
        raise ValueError("No scales provided.")
    return scales


def scale_label(scale: float) -> str:
    # 0.99 -> s0990, 1.01 -> s1010
    return f"s{int(round(scale * 1000)):04d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create globally scaled submission files.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input submission CSV, e.g. submissions/submission_feature_rich_blend_refined5.csv",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default="0.98,0.99,1.01,1.02",
        help="Comma-separated scale factors.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="submissions",
        help="Directory to write scaled submissions.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="submission_refined5_global_scale",
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--clip-min",
        type=float,
        default=0.0,
        help="Clip Revenue to at least this value after scaling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    if "Revenue" not in df.columns:
        raise ValueError("Input CSV must contain a Revenue column.")

    scales = parse_scales(args.scales)
    summary_rows = []

    for scale in scales:
        out = df.copy()
        out["Revenue"] = (out["Revenue"].astype(float) * float(scale)).clip(lower=float(args.clip_min))

        out_path = output_dir / f"{args.prefix}_{scale_label(scale)}.csv"
        out.to_csv(out_path, index=False)

        summary_rows.append(
            {
                "scale": scale,
                "output": str(out_path),
                "min": float(out["Revenue"].min()),
                "max": float(out["Revenue"].max()),
                "mean": float(out["Revenue"].mean()),
                "std": float(out["Revenue"].std()),
                "sum": float(out["Revenue"].sum()),
            }
        )

        print(f"Wrote {out_path}")

    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / f"{args.prefix}_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nSummary:")
    print(summary.to_string(index=False))
    print(f"\nWrote summary: {summary_path}")


if __name__ == "__main__":
    main()
