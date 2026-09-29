from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts/foundation_results_to_md.py"
    spec = importlib.util.spec_from_file_location("foundation_results_to_md", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_markdown_reads_each_metrics_file(tmp_path):
    output = tmp_path / "one"
    output.mkdir()
    (output / "metrics.json").write_text(
        json.dumps(
            {
                "metrics_original_power_units": {
                    "count": 2,
                    "mae": 1.25,
                    "rmse": 2.5,
                    "nmae_percent": 3.0,
                    "nrmse_percent": 4.0,
                    "mape_percent": None,
                }
            }
        ),
        encoding="utf-8",
    )
    summary = tmp_path / "run_summary.tsv"
    summary.write_text(
        "seq_len\tpred_len\tdataset\tmodel\tmode\tstatus\toutput_dir\texit_code\tlaunch_gpu\tdata_fingerprint\n"
        f"48\t1\tskippd_luoyang\tSundial\tzero_shot\tPASS\t{output}\t0\t0\tfoundation-data-v1:abc\n",
        encoding="utf-8",
    )

    rendered = _module().build_markdown(summary)

    assert "Tasks: 1" in rendered
    assert "Passed: 1" in rendered
    assert "| skippd_luoyang | 48 | 1 | Sundial | zero_shot | PASS |" in rendered
    assert "1.25" in rendered
    assert "2.5" in rendered
