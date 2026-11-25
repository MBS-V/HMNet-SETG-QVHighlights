import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.setg.parser import SubEventParser
p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--output',default='outputs/query_tags.csv'); p.add_argument('--max-subevents',type=int,default=4); a=p.parse_args()
parser=SubEventParser(a.max_subevents); rows=[]
for i,line in enumerate(Path(a.manifest).read_text().splitlines()):
 if not line.strip(): continue
 r=json.loads(line); z=parser.parse(r.get('query','')); rows.append({'qid':r.get('qid',i),'query':z.query,'K':z.num_subevents,'bucket':'simple' if z.num_subevents==1 else 'compound','status':z.status,'subevents':' || '.join(z.subevents),'warnings':' | '.join(z.warnings)})
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
with open(a.output,'w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys() if rows else ['qid','query','K','bucket','status','subevents','warnings']); w.writeheader(); w.writerows(rows)
print(f'Wrote {len(rows)} parser tags to {a.output}')
