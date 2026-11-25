import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation.comparison import difference_in_differences
from src.evaluation.bootstrap import bootstrap_difference_in_differences
p=argparse.ArgumentParser(); p.add_argument('--input',required=True,help='JSON with four means and optionally four per-query arrays'); p.add_argument('--output',default='outputs/comparison.json'); p.add_argument('--resamples',type=int,default=1000); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
r=json.loads(Path(a.input).read_text()); out=difference_in_differences(r['hmnet_simple'],r['setg_simple'],r['hmnet_compound'],r['setg_compound'])
keys=['hmnet_simple_per_query','setg_simple_per_query','hmnet_compound_per_query','setg_compound_per_query']
if all(k in r for k in keys):
    hs,ss,hc,sc=[r[k] for k in keys]
    out=difference_in_differences(sum(hs)/len(hs),sum(ss)/len(ss),sum(hc)/len(hc),sum(sc)/len(sc))
    out['bootstrap_ci']=bootstrap_difference_in_differences(hs,ss,hc,sc,a.resamples,0.95,a.seed)
Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2))
md=['# HM-Net vs SE-TG comparison','',f"- Simple-query improvement: {out['simple_improvement']:.6f}",f"- Compound-query improvement: {out['compound_improvement']:.6f}",f"- Difference-in-differences: {out['difference_in_differences']:.6f}"]
if 'bootstrap_ci' in out:
 c=out['bootstrap_ci']; md += ['',f"- {c['confidence']:.0%} bootstrap CI: [{c['lower']:.6f}, {c['upper']:.6f}]",f"- Includes zero: {c['includes_zero']}",f"- Resamples: {c['n_resamples']}; simple n={c['simple_queries']}; compound n={c['compound_queries']}"]
Path(a.output).with_suffix('.md').write_text('\n'.join(md)+'\n'); print(json.dumps(out,indent=2))
