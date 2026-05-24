#!/usr/bin/env python3
"""Export common TensorBoard scalar plots for two runs.

Outputs:
- One PNG per common scalar tag (both runs overlaid)
- One combined multi-page PDF with all common tags
- tags.txt listing exported tags

Usage:
/isaac-sim/python.sh scripts/export_tb_compare.py \
  --run-a "logs/skrl/jettank_direct/<run_a>" \
  --run-b "logs/skrl/jettank_direct/<run_b>" \
  --label-a "<label_a>" \
  --label-b "<label_b>" \
  --out "outputs/tb-compare-<name>"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _safe_name(tag: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", tag).strip("_")


def _load_scalars(event_file: Path) -> dict[str, tuple[list[int], list[float]]]:
    acc = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    acc.Reload()
    result: dict[str, tuple[list[int], list[float]]] = {}
    for tag in acc.Tags().get("scalars", []):
        events = acc.Scalars(tag)
        steps = [e.step for e in events]
        vals = [float(e.value) for e in events]
        result[tag] = (steps, vals)
    return result


def _resolve_event_file(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("events.out.tfevents.*"))
    if not candidates:
        raise FileNotFoundError(f"No TensorBoard event files in: {run_dir}")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export shared TensorBoard scalar plots for two runs")
    parser.add_argument("--run-a", required=True, help="Path to run A directory")
    parser.add_argument("--run-b", required=True, help="Path to run B directory")
    parser.add_argument("--label-a", default="run_a", help="Legend label for run A")
    parser.add_argument("--label-b", default="run_b", help="Legend label for run B")
    parser.add_argument("--out", required=True, help="Output directory for exported plots")
    parser.add_argument("--dpi", type=int, default=180, help="PNG dpi")
    parser.add_argument("--max-plots", type=int, default=0, help="Limit number of exported tags (0 = all)")
    args = parser.parse_args()

    run_a_dir = Path(args.run_a).expanduser().resolve()
    run_b_dir = Path(args.run_b).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    event_a = _resolve_event_file(run_a_dir)
    event_b = _resolve_event_file(run_b_dir)

    data_a = _load_scalars(event_a)
    data_b = _load_scalars(event_b)

    common_tags = sorted(set(data_a.keys()) & set(data_b.keys()))
    if args.max_plots > 0:
        common_tags = common_tags[: args.max_plots]

    if not common_tags:
        raise RuntimeError("No common scalar tags found between runs")

    pdf_path = out_dir / "all_common_scalars.pdf"
    with PdfPages(pdf_path) as pdf:
        for tag in common_tags:
            steps_a, vals_a = data_a[tag]
            steps_b, vals_b = data_b[tag]

            fig = plt.figure(figsize=(8, 4.5))
            ax = fig.add_subplot(111)
            ax.plot(steps_a, vals_a, label=args.label_a, linewidth=1.6)
            ax.plot(steps_b, vals_b, label=args.label_b, linewidth=1.6)
            ax.set_title(tag)
            ax.set_xlabel("Step")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()

            png_path = out_dir / f"{_safe_name(tag)}.png"
            fig.savefig(png_path, dpi=args.dpi)
            pdf.savefig(fig)
            plt.close(fig)

    (out_dir / "tags.txt").write_text("\n".join(common_tags) + "\n", encoding="utf-8")
    print(f"Run A event: {event_a}")
    print(f"Run B event: {event_b}")
    print(f"Common scalar tags: {len(common_tags)}")
    print(f"PNG dir: {out_dir}")
    print(f"Combined PDF: {pdf_path}")


if __name__ == "__main__":
    main()
