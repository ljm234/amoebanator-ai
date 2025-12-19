import json
from pathlib import Path
import numpy as np

MET = Path("outputs/metrics")
MOD = Path("outputs/model")

def _load_T():
    for p in [MOD/"temperature_scale.json", MET/"temperature_scale.json"]:
        if p.exists():
            return float(json.loads(p.read_text())["T"])
    raise FileNotFoundError("temperature_scale.json not found")

def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)

def main():
    logits = np.load(MET/"val_logits.npy")
    y = np.load(MET/"val_y.npy").astype(int)
    T = _load_T()
    p = _softmax(logits / T)

    conf = json.loads((MET/"conformal.json").read_text())
    qhat = float(conf["qhat"]); alpha = float(conf["alpha"])

    nonconf = 1.0 - p[np.arange(len(y)), y]
    coverage = float((nonconf <= qhat).mean())

    p_high = p[:,1]
    include_high = p_high >= (1.0 - qhat)
    include_low  = (1.0 - p_high) >= (1.0 - qhat)
    set_size = include_high.astype(int) + include_low.astype(int)
    ambig = float((set_size == 2).mean())
    single = float((set_size == 1).mean())
    empty = float((set_size == 0).mean())

    out = {
      "alpha": alpha,
      "target_coverage": 1.0 - alpha,
      "empirical_coverage": coverage,
      "set_size_singleton": single,
      "set_size_ambiguous": ambig,
      "set_size_empty": empty
    }
    (MET/"conformal_eval.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
