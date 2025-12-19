from __future__ import annotations
import sys, json
from pathlib import Path
import pandas as pd

from ml.infer import infer_one
from ml.robust.ood import STATS_JSON

LOG_CSV = Path("outputs/diagnosis_log_pro.csv")

def _load_stats_cols() -> list[str]:
    if Path(STATS_JSON).exists():
        obj = json.loads(Path(STATS_JSON).read_text())
        cols = obj.get("cols", [])
        bonus = ["age","csf_glucose","csf_protein","csf_wbc","pcr","microscopy","exposure","risk_score","symptoms"]
        seen = set(cols)
        cols += [c for c in bonus if c not in seen]
        return cols
    return ["age","csf_glucose","csf_protein","csf_wbc","pcr","microscopy","exposure","risk_score","symptoms"]

def main():
    use_last = ("--last" in sys.argv)
    json_arg = None
    if "--json" in sys.argv:
        i = sys.argv.index("--json")
        if i+1 < len(sys.argv):
            json_arg = sys.argv[i+1]

    if not use_last and not json_arg:
        print("Provide --last or --json '{...}'", file=sys.stderr)
        sys.exit(1)

    if json_arg:
        row = pd.Series(json.loads(json_arg))
    else:
        if not LOG_CSV.exists():
            print("Missing outputs/diagnosis_log_pro.csv", file=sys.stderr)
            sys.exit(2)
        df = pd.read_csv(LOG_CSV)
        if df.empty:
            print("Empty outputs/diagnosis_log_pro.csv", file=sys.stderr)
            sys.exit(3)
        want = _load_stats_cols()
        have = [c for c in want if c in df.columns]
        row = df.iloc[-1].reindex(have)

    out = infer_one(row)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
