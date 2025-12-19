import numpy as np
import pandas as pd

def decision_curve(y_true, p, thresholds=np.linspace(0.01,0.99,99)):
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)
    N = len(y_true)
    prevalence = y_true.mean() if N>0 else 0.0
    out = []
    for t in thresholds:
        treat = (p >= t).astype(int)
        tp = ((treat==1) & (y_true==1)).sum()
        fp = ((treat==1) & (y_true==0)).sum()
        net_benefit = (tp/N) - (fp/N) * (t/(1-t))
        out.append((float(t), float(net_benefit)))
    return pd.DataFrame(out, columns=["threshold","net_benefit"])
