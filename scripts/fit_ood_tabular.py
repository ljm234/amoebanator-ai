import json
from argparse import ArgumentParser
from ml.robust.ood import fit_tabular_stats, STATS_JSON

def main():
    p = ArgumentParser()
    p.add_argument("--quantile", type=float, default=0.999)
    p.add_argument("--drop-cols", type=str, default="")
    p.add_argument("--full", action="store_true")
    args = p.parse_args()
    drops = [c.strip() for c in args.drop_cols.split(",") if c.strip()]
    out = fit_tabular_stats(quantile=args.quantile, drop_cols=drops, use_diagonal=not args.full)
    if "numeric_cols" not in out:
        out["numeric_cols"] = out.get("cols", [])
    if "cov" not in out:
        S = out.get("S", [])
        n = len(out.get("cols", []))
        if isinstance(S, list) and len(S) > 0 and isinstance(S[0], list):
            out["cov"] = S
        else:
            mat = [[0.0]*n for _ in range(n)]
            for i in range(min(n, len(S))):
                mat[i][i] = float(S[i])
            out["cov"] = mat
    STATS_JSON.write_text(json.dumps(out, indent=2))
    tau = float(out.get("tau", float("nan")))
    print(f"Saved outputs/metrics/feature_stats.json with {len(out.get('cols', []))} cols; tau={tau:.3f}; quantile={args.quantile}")

if __name__ == "__main__":
    main()
