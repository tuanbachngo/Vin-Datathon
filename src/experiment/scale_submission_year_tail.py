"""Create year-specific and tail-horizon scaled submission files.

Use this after finding that global scale 1.04 improves your refined5 submission.

Recommended input:
    submissions/submission_feature_rich_blend_refined5.csv

Year scaling formula:
    2023 rows: Revenue * scale_2023
    2024 rows: Revenue * scale_2024

Tail scaling formula:
    horizon_day < tail_start_day:  Revenue * early_scale
    horizon_day >= tail_start_day: Revenue * tail_scale

Examples:
    python scale_submission_year_tail.py --input "submissions/submission_feature_rich_blend_refined5.csv" --mode both

    python scale_submission_year_tail.py --input "submissions/submission_feature_rich_blend_refined5.csv" --mode year --year-pairs "1.04:1.02,1.04:1.03,1.04:1.05,1.04:1.06"

    python scale_submission_year_tail.py --input "submissions/submission_feature_rich_blend_refined5.csv" --mode tail --tail-start-day 366 --tail-pairs "1.04:1.02,1.04:1.03,1.04:1.05,1.04:1.06"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _parse_pairs(text: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid pair {part!r}; expected format a:b")
        left, right = part.split(":", 1)
        pairs.append((float(left.strip()), float(right.strip())))
    if not pairs:
        raise ValueError("No scale pairs provided.")
    return pairs


def _scale_label(value: float) -> str:
    # 1.04 -> 1040, 1.034 -> 1034
    return f"{int(round(value * 1000)):04d}"


def _load_submission(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path, parse_dates=["Date"])
    if "Date" not in df.columns or "Revenue" not in df.columns:
        raise ValueError("Input CSV must contain Date and Revenue columns.")
    return df


def _write_summary(rows: list[dict], output_dir: Path, prefix: str) -> None:
    if not rows:
        return
    summary = pd.DataFrame(rows)
    summary_path = output_dir / f"{prefix}_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\nSummary:")
    print(summary.to_string(index=False))
    print(f"\nWrote summary: {summary_path}")


def write_year_scaled(
    df: pd.DataFrame,
    *,
    pairs: list[tuple[float, float]],
    output_dir: Path,
    prefix: str,
    clip_min: float,
) -> list[dict]:
    rows: list[dict] = []
    base = df.copy()
    base["Date"] = pd.to_datetime(base["Date"])

    for scale_2023, scale_2024 in pairs:
        out = base.copy()
        scale = pd.Series(1.0, index=out.index, dtype=float)
        scale.loc[out["Date"].dt.year == 2023] = float(scale_2023)
        scale.loc[out["Date"].dt.year == 2024] = float(scale_2024)
        out["Revenue"] = (out["Revenue"].astype(float) * scale).clip(lower=float(clip_min))

        out_path = output_dir / (
            f"{prefix}_year_y2023s{_scale_label(scale_2023)}"
            f"_y2024s{_scale_label(scale_2024)}.csv"
        )
        out.to_csv(out_path, index=False)
        rows.append(
            {
                "mode": "year",
                "scale_2023": scale_2023,
                "scale_2024": scale_2024,
                "tail_start_day": "",
                "early_scale": "",
                "tail_scale": "",
                "output": str(out_path),
                "min": float(out["Revenue"].min()),
                "max": float(out["Revenue"].max()),
                "mean": float(out["Revenue"].mean()),
                "std": float(out["Revenue"].std()),
                "sum": float(out["Revenue"].sum()),
            }
        )
        print(f"Wrote {out_path}")

    return rows


def write_tail_scaled(
    df: pd.DataFrame,
    *,
    pairs: list[tuple[float, float]],
    tail_start_day: int,
    output_dir: Path,
    prefix: str,
    clip_min: float,
) -> list[dict]:
    rows: list[dict] = []
    base = df.copy()
    base["Date"] = pd.to_datetime(base["Date"])

    # Preserve original row order in the output, but define horizon days by sorted Date.
    sorted_idx = base.sort_values("Date").index.to_list()
    horizon_day = pd.Series(index=base.index, dtype=int)
    horizon_day.loc[sorted_idx] = range(1, len(base) + 1)

    for early_scale, tail_scale in pairs:
        out = base.copy()
        scale = pd.Series(float(early_scale), index=out.index, dtype=float)
        scale.loc[horizon_day >= int(tail_start_day)] = float(tail_scale)
        out["Revenue"] = (out["Revenue"].astype(float) * scale).clip(lower=float(clip_min))

        out_path = output_dir / (
            f"{prefix}_tail_d{int(tail_start_day)}"
            f"_earlys{_scale_label(early_scale)}"
            f"_tails{_scale_label(tail_scale)}.csv"
        )
        out.to_csv(out_path, index=False)
        rows.append(
            {
                "mode": "tail",
                "scale_2023": "",
                "scale_2024": "",
                "tail_start_day": int(tail_start_day),
                "early_scale": early_scale,
                "tail_scale": tail_scale,
                "output": str(out_path),
                "min": float(out["Revenue"].min()),
                "max": float(out["Revenue"].max()),
                "mean": float(out["Revenue"].mean()),
                "std": float(out["Revenue"].std()),
                "sum": float(out["Revenue"].sum()),
            }
        )
        print(f"Wrote {out_path}")

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create year/tail scaled submission files.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input submission CSV, usually submissions/submission_feature_rich_blend_refined5.csv",
    )
    parser.add_argument(
        "--mode",
        choices=["year", "tail", "both"],
        default="both",
        help="Which scaling variants to create.",
    )
    parser.add_argument(
        "--year-pairs",
        type=str,
        default="1.04:1.02,1.04:1.03,1.04:1.04,1.04:1.05,1.04:1.06",
        help="Comma-separated 2023:2024 scale pairs.",
    )
    parser.add_argument(
        "--tail-pairs",
        type=str,
        default="1.04:1.02,1.04:1.03,1.04:1.04,1.04:1.05,1.04:1.06",
        help="Comma-separated early:tail scale pairs.",
    )
    parser.add_argument(
        "--tail-start-day",
        type=int,
        default=366,
        help="First horizon day receiving tail_scale in tail mode.",
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
        default="submission_refined5_year_tail_scale",
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--clip-min",
        type=float,
        default=0.0,
        help="Minimum Revenue after scaling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_submission(input_path)

    all_rows: list[dict] = []

    if args.mode in {"year", "both"}:
        print("\n=== Year scaling ===")
        all_rows.extend(
            write_year_scaled(
                df,
                pairs=_parse_pairs(args.year_pairs),
                output_dir=output_dir,
                prefix=args.prefix,
                clip_min=float(args.clip_min),
            )
        )

    if args.mode in {"tail", "both"}:
        print("\n=== Tail scaling ===")
        all_rows.extend(
            write_tail_scaled(
                df,
                pairs=_parse_pairs(args.tail_pairs),
                tail_start_day=int(args.tail_start_day),
                output_dir=output_dir,
                prefix=args.prefix,
                clip_min=float(args.clip_min),
            )
        )

    _write_summary(all_rows, output_dir, args.prefix)


if __name__ == "__main__":
    main()
