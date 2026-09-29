#!/usr/bin/env python3
"""Convert a foundation-model TSV run summary and metrics into Markdown."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


METRIC_COLUMNS = (
    ("count", "Count"),
    ("mae", "MAE"),
    ("rmse", "RMSE"),
    ("nmae_percent", "NMAE (%)"),
    ("nrmse_percent", "NRMSE (%)"),
    ("mape_percent", "MAPE (%)"),
)


def _format(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return "-" if value != value else f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def build_markdown(summary_path: Path) -> str:
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows: List[Dict[str, str]] = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError("run summary contains no experiment rows")

    rendered = []
    for row in rows:
        output_dir = Path(row["output_dir"])
        metrics_path = output_dir / "metrics.json"
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError) as exc:
            raise ValueError(f"cannot read metrics: {metrics_path}") from exc
        original = metrics.get("metrics_original_power_units", {})
        if not isinstance(original, dict):
            raise ValueError(f"invalid original-power metrics: {metrics_path}")
        rendered.append((row, original))

    pass_count = sum(row.get("status") == "PASS" for row, _ in rendered)
    lines = [
        "# Foundation-model experiment results",
        "",
        f"- Tasks: {len(rendered)}",
        f"- Passed: {pass_count}",
        f"- Failed: {len(rendered) - pass_count}",
        f"- Source summary: `{summary_path}`",
        "",
        "| Dataset | Seq | Pred | Model | Mode | Status | "
        + " | ".join(label for _, label in METRIC_COLUMNS)
        + " |",
        "|---|---:|---:|---|---|---|"
        + "---:|" * len(METRIC_COLUMNS),
    ]
    for row, original in rendered:
        values = [
            row.get("dataset", ""),
            row.get("seq_len", ""),
            row.get("pred_len", ""),
            row.get("model", ""),
            row.get("mode", ""),
            row.get("status", ""),
        ]
        values.extend(_format(original.get(key)) for key, _ in METRIC_COLUMNS)
        lines.append("| " + " | ".join(_format(value) for value in values) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    markdown = build_markdown(args.summary.resolve())
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(markdown, encoding="utf-8", newline="\n")
    temporary.replace(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
