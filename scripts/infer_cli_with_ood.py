#!/usr/bin/env python3
import sys, json, argparse
from pathlib import Path
import pandas as pd

from ml.infer import infer_one
from ml.ood import ood_abstain_from_p
from ml.ood_energy import ood_abstain_energy

LOG_PATH = Path("outputs/diagnosis_log_pro.csv")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", action="store_true")
    ap.add_argument("--json", type=str)
    args = ap.parse_args()

    if args.last:
        if not LOG_PATH.exists():
            print("Missing outputs/diagnosis_log_pro.csv", file=sys.stderr); sys.exit(1)
        df = pd.read_csv(LOG_PATH)
        if df.empty:
            print("Log CSV is empty", file=sys.stderr); sys.exit(1)
        row = df.iloc[-1].to_dict()
    elif args.json:
        row = json.loads(args.json)
    else:
        print("Provide --last or --json '{...}'", file=sys.stderr); sys.exit(1)

    base = infer_one(row)               # conformal (include_low/high) + p_high + threshold
    ent  = ood_abstain_from_p(base["p_high"])       # entropy gate
    eng  = ood_abstain_energy(base["p_high"])       # energy gate

    abstain_conformal = bool(base.get("include_low", False) and base.get("include_high", False))
    abstain_final = bool(abstain_conformal or ent["ood_abstain"] or eng["ood_abstain_energy"])

    out = {
        "prediction": base.get("prediction"),
        "p_high": float(base.get("p_high")),
        "threshold": float(base.get("threshold")),
        "qhat": float(base.get("qhat")),
        "conformal_include_low": bool(base.get("include_low", False)),
        "conformal_include_high": bool(base.get("include_high", False)),
        "entropy": float(ent["entropy"]), "tau_entropy": ent["tau"],
        "energy_neg": float(eng["energy_neg"]), "tau_energy_neg": eng["tau"],
        "abstain_conformal": abstain_conformal,
        "abstain_entropy": bool(ent["ood_abstain"]),
        "abstain_energy": bool(eng["ood_abstain_energy"]),
        "final_decision": ("ABSTAIN" if abstain_final else base.get("prediction"))
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
