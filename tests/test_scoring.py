import pandas as pd
from pathlib import Path
import subprocess, json, os

def test_log_headers_exist():
    from app import load_log
    df = load_log("outputs/diagnosis_log_pro.csv")
    expected = ["timestamp_tz","case_id","source","physician","age","sex","csf_glucose","csf_protein","csf_wbc","symptoms","pcr","microscopy","exposure","risk_score","risk_label","comments"]
    for c in expected:
        assert c in df.columns
