from typing import Tuple

LABELS = ("Low", "High")

def set_from_p_high(p_high: float, qhat: float) -> Tuple[bool, bool]:
    include_high = p_high >= (1.0 - qhat)
    include_low  = p_high <= qhat
    return (include_low, include_high)

def decision_from_p_high(p_high: float, qhat: float) -> str:
    low, high = set_from_p_high(p_high, qhat)
    if low and high:
        return "ABSTAIN"
    return "High" if high else "Low"
