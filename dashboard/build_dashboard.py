#!/usr/bin/env python3
"""Regenerates dashboard/index.html from template.html + the real results/*.csv
files, exactly mirroring how paper/build_paper.py builds the paper from the
same data. Run this after re-running any experiments/run_*.py script so the
dashboard's embedded numbers stay in sync with the source CSVs.
"""
import json
from pathlib import Path
import pandas as pd

R = Path(__file__).resolve().parent.parent
D = Path(__file__).resolve().parent

def records(name):
    return json.loads(pd.read_csv(R / "results" / f"{name}.csv").to_json(orient="records"))

data = {
    "stat": records("statistical_validation"),
    "sig": records("significance_tests"),
    "abl": records("ablation_study"),
    "timing": records("timing"),
    "diag": records("diagnosis_validation"),
    "diag_acc": records("diagnosis_accuracy"),
}

template = (D / "template.html").read_text()
out = template.replace("__DATA_JSON__", json.dumps(data, separators=(",", ":")))
(D / "index.html").write_text(out)
print(f"Saved {D / 'index.html'} ({len(out):,} bytes)")

# GitHub Pages only serves from "/" or "/docs", so docs/ is a required mirror
# of dashboard/index.html, not a separate document set -- keep them identical.
docs_dir = R / "docs"
docs_dir.mkdir(exist_ok=True)
(docs_dir / "index.html").write_text(out)
print(f"Saved {docs_dir / 'index.html'} (GitHub Pages mirror)")
