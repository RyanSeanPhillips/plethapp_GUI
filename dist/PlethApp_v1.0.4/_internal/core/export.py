from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict

def build_dataframe(t, y, peaks_idx=None, sweep_idx=0, meta: Dict = None) -> pd.DataFrame:
    df = pd.DataFrame({"t": t, "y": y})
    df["sweep"] = sweep_idx
    if peaks_idx is not None:
        df["is_peak"] = False
        df.loc[peaks_idx, "is_peak"] = True
    if meta:
        for k, v in meta.items():
            df[k] = v
    return df

def save_outputs(out_path: Path, df: pd.DataFrame, summary_text: str = ""):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    (out_path.with_suffix(".txt")).write_text(summary_text or "Summary TBD\n", encoding="utf-8")

def generate_summary_pdf(in_path: Path, out_path: Path):
    # TODO: call your existing summary_report pipeline
    pass
