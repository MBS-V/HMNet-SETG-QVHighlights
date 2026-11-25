import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.evaluation.duration_matching import duration_match_indices
p=argparse.ArgumentParser(); p.add_argument('--tags',required=True); p.add_argument('--manifest',required=True); p.add_argument('--output',default='outputs/duration_match_report.json'); a=p.parse_args()
tags={r['qid']:r for r in csv.DictReader(open(a.tags,encoding='utf-8'))}; rows=[json.loads(s) for s in Path(a.manifest).read_text().splitlines() if s.strip()]
simple=[]; compound=[]
for i,r in enumerate(rows):
 tag=tags.get(str(r.get('qid',i)))
 if not tag: continue
 windows=r.get('timestamps',r.get('relevant_windows'))
 intervals=sorted((float(w[0]),float(w[1])) for w in (windows if isinstance(windows[0],(list,tuple)) else [windows]))
 merged=[]
 for lo,hi in intervals:
  if merged and lo<=merged[-1][1]: merged[-1]=(merged[-1][0],max(merged[-1][1],hi))
  else: merged.append((lo,hi))
 item={'qid':str(r.get('qid',i)),'duration':sum(hi-lo for lo,hi in merged)}
 (simple if tag['bucket']=='simple' else compound).append(item)
si,ci,report=duration_match_indices([x['duration'] for x in simple],[x['duration'] for x in compound])
report.update({'simple_total':len(simple),'compound_total':len(compound),'matched_simple_qids':[simple[i]['qid'] for i in si],'matched_compound_qids':[compound[i]['qid'] for i in ci]})
Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
