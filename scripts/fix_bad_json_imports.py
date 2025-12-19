import pathlib, re
for p in pathlib.Path('.').rglob('*.py'):
    try:
        s = p.read_text(errors='ignore')
    except Exception:
        continue
    s2 = re.sub(r'from pathlib import Path,\s*json', 'from pathlib import Path\nimport json', s)
    s2 = re.sub(r'from pathlib import json', 'import json', s2)
    if s2 != s:
        p.write_text(s2)
        print(p)
