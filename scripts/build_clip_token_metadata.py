"""Build exact CLIP token IDs, attention masks and character offsets for query strings.

Requires transformers with the fast OpenAI CLIP tokenizer. Tokenization is separate
from feature extraction and does not modify the precomputed feature arrays.
"""
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--annotations',required=True); p.add_argument('--output',required=True)
    p.add_argument('--tokenizer',default='openai/clip-vit-base-patch32'); p.add_argument('--max-length',type=int,default=77); a=p.parse_args()
    try: from transformers import CLIPTokenizerFast
    except ImportError as e: raise SystemExit('Install optional tokenizer dependencies: pip install -r requirements-tokenizer.txt') from e
    tokenizer=CLIPTokenizerFast.from_pretrained(a.tokenizer)
    rows=[]
    for line_no,line in enumerate(Path(a.annotations).read_text(encoding='utf-8').splitlines(),1):
        if not line.strip(): continue
        ann=json.loads(line); encoded=tokenizer(ann['query'],padding='max_length',truncation=True,max_length=a.max_length,return_offsets_mapping=True)
        rows.append({'qid':str(ann['qid']),'token_ids':encoded['input_ids'],'text_mask':encoded['attention_mask'],
                     'token_offsets':[list(x) for x in encoded['offset_mapping']],
                     'mlm_mask':[int(m and end>start) for m,(start,end) in zip(encoded['attention_mask'],encoded['offset_mapping'])]})
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text('\n'.join(json.dumps(x) for x in rows)+'\n',encoding='utf-8')
    print(f'Wrote tokenizer metadata for {len(rows)} queries to {out}')

if __name__=='__main__': main()
