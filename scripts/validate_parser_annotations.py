"""Calculate Cohen's kappa and parser agreement from the manual CSV template."""
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation.parser_validation import agreement
p=argparse.ArgumentParser(); p.add_argument('--csv',required=True); p.add_argument('--output',default='outputs/parser_agreement.json'); a=p.parse_args()
rows=[r for r in csv.DictReader(open(a.csv,encoding='utf-8')) if r.get('annotator_a','').strip() and r.get('annotator_b','').strip()]
def label(x):
 x=x.strip().lower()
 if x in {'simple','0'}: return 0
 if x in {'compound','1'}: return 1
 raise ValueError(f'Unknown label {x!r}; use simple/compound')
A=[label(r['annotator_a']) for r in rows]; B=[label(r['annotator_b']) for r in rows]; P=[label(r['parser_label']) for r in rows if r.get('parser_label','').strip()]
if len(P)!=len(rows): P=None
out=agreement(A,B,P); Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
