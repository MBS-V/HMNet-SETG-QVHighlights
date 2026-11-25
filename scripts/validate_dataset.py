"""Validate a manifest. Prints a short summary; the full report (incl. every missing file) goes to --output."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.feature_validation import validate_manifest
p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--feature-root')
p.add_argument('--output',default='outputs/data_validation.json'); p.add_argument('--max-rows',type=int); a=p.parse_args()
r=validate_manifest(a.manifest,a.feature_root,a.max_rows)
Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(r,indent=2))
print(f"{a.manifest}: {r['loadable_rows']}/{r['samples']} rows loadable | video dims {r['video_dims']} | text dims {r['text_dims']} | video length {r['video_length_min']}-{r['video_length_max']}")
print('info:',r['info'])
for cat,n in sorted(r['error_counts'].items(),key=lambda x:-x[1]):
    print(f'  ERROR {cat}: {n}   e.g. {r["error_examples"][cat][0]}')
print('VALID' if r['valid'] else f'NOT VALID - full report: {a.output}')
sys.exit(0 if r['valid'] else 2)
