"""JSONL annotation manifest parsing and basic temporal validation."""
import json
from pathlib import Path

def read_jsonl(path):
    rows=[]
    for line_no,line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(),1):
        if not line.strip(): continue
        try: rows.append(json.loads(line))
        except json.JSONDecodeError as exc: raise ValueError(f'{path}:{line_no}: {exc}') from exc
    return rows

def validate_annotation(row):
    required=('query','duration','timestamps','video_feature','text_feature')
    missing=[k for k in required if k not in row]
    if missing: raise ValueError(f'Missing required annotation fields: {missing}')
    d=float(row['duration']); s,e=map(float,row['timestamps'])
    if not d>0 or not 0<=s<e<=d+1e-3: raise ValueError(f'Invalid temporal annotation: duration={d}, timestamps={(s,e)}')
    return True
