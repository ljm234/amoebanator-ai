from pathlib import Path
import json

p = Path("outputs/metrics/feature_stats.json")
if p.exists():
    d = json.loads(p.read_text())
    cols = d.get("numeric_cols") or d.get("cols") or []
    n = len(cols)
    cov = d.get("cov")
    if not (isinstance(cov, list) and cov and isinstance(cov[0], list) and len(cov) == n and len(cov[0]) == n):
        S = d.get("S", [])
        if isinstance(S, list) and S:
            if isinstance(S[0], list):
                try:
                    M = [[float(val) for val in row[:n]] for row in S[:n]]
                    if len(M) == n and len(M[0]) == n:
                        d["cov"] = M
                except Exception:
                    pass
            else:
                M = [[0.0] * n for _ in range(n)]
                for i, v in enumerate(S[:n]):
                    try:
                        M[i][i] = float(v)
                    except Exception:
                        try:
                            M[i][i] = float(v[0])
                        except Exception:
                            M[i][i] = 0.0
                d["cov"] = M
        else:
            d["cov"] = [[0.0] * n for _ in range(n)]
    p.write_text(json.dumps(d, indent=2))
    print(len(d.get("cov", [])))
else:
    print(0)
