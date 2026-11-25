"""Normalize official QVHighlights JSONL + HDF5 feature stores into this repo's manifest.

No data is downloaded. The script validates the supplied HDF5 keys and preserves
all relevant_windows rather than silently selecting only the first moment.
"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.feature_loader import load_array,validate_feature_array

def rel_or_abs(path,root):
    p=Path(path).resolve(); r=Path(root).resolve()
    try: return str(p.relative_to(r))
    except ValueError: return str(p)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--annotations',required=True,help='Official QVHighlights train/val JSONL')
    p.add_argument('--feature-root',required=True)
    p.add_argument('--slowfast-h5',required=True)
    p.add_argument('--clip-video-h5',required=True)
    p.add_argument('--clip-text-h5',required=True)
    p.add_argument('--text-key-field',choices=['qid','vid'],default='qid')
    p.add_argument('--token-metadata',help='Optional JSONL with qid, token_ids and token_offsets from the exact CLIP tokenizer')
    p.add_argument('--output',required=True)
    a=p.parse_args(); root=Path(a.feature_root).resolve()
    paths={k:rel_or_abs(v,root) for k,v in {'slowfast':a.slowfast_h5,'clip_video':a.clip_video_h5,'clip_text':a.clip_text_h5}.items()}
    token_meta={}
    if a.token_metadata:
        for line in Path(a.token_metadata).read_text(encoding='utf-8').splitlines():
            if line.strip():
                x=json.loads(line); token_meta[str(x['qid'])]=x
    rows=[]; errors=[]
    for line_no,line in enumerate(Path(a.annotations).read_text(encoding='utf-8').splitlines(),1):
        if not line.strip(): continue
        try:
            ann=json.loads(line); qid=str(ann['qid']); vid=str(ann['vid']); query=ann['query']
            refs=[{'path':paths['slowfast'],'key':vid},{'path':paths['clip_video'],'key':vid}]
            arrays=[load_array(root/ref['path'] if not Path(ref['path']).is_absolute() else ref['path'],ref['key']) for ref in refs]
            shapes=[validate_feature_array(x,'video modality') for x in arrays]
            if len({s[0] for s in shapes})!=1: raise ValueError(f'video modality sequence lengths differ: {shapes}')
            text_key=str(ann[a.text_key_field])
            text_path=paths['clip_text']
            text_arr=load_array(root/text_path if not Path(text_path).is_absolute() else text_path,text_key)
            text_shape=validate_feature_array(text_arr,'text feature')
            windows=ann.get('relevant_windows')
            if not windows: raise ValueError('training/validation annotation has no relevant_windows')
            duration=float(ann['duration'])
            if duration<=0 or any(not (0<=float(w[0])<float(w[1])<=duration+1e-3) for w in windows):
                raise ValueError(f'invalid relevant_windows for duration={duration}: {windows}')
            saliency=None
            if 'saliency_scores' in ann and 'relevant_clip_ids' in ann:
                saliency=[[-1.0,-1.0,-1.0] for _ in range(shapes[0][0])]
                if len(ann['saliency_scores'])!=len(ann['relevant_clip_ids']): raise ValueError('saliency_scores/relevant_clip_ids length mismatch')
                for clip_id,ratings in zip(ann['relevant_clip_ids'],ann['saliency_scores']):
                    clip_id=int(clip_id)
                    if clip_id<0 or clip_id>=len(saliency): raise ValueError(f'relevant clip id {clip_id} outside feature sequence length {len(saliency)}')
                    saliency[clip_id]=[float(x) for x in ratings]
            meta=token_meta.get(qid,{})
            row={'qid':qid,'video_id':vid,'query':query,'duration':duration,
                 'relevant_windows':ann['relevant_windows'],'video_features':refs,
                 'text_feature':text_path,'text_key':text_key,'saliency':saliency}
            if 'token_ids' in meta: row['token_ids']=meta['token_ids']
            if 'token_offsets' in meta: row['token_offsets']=meta['token_offsets']
            if 'text_mask' in meta: row['text_mask']=meta['text_mask']
            if 'mlm_mask' in meta: row['mlm_mask']=meta['mlm_mask']
            if row.get('token_ids') is not None and len(row['token_ids'])!=text_shape[0]: raise ValueError('token_ids length differs from CLIP text feature length')
            if row.get('token_offsets') is not None and len(row['token_offsets'])!=text_shape[0]: raise ValueError('token_offsets length differs from CLIP text feature length')
            if row.get('text_mask') is not None and len(row['text_mask'])!=text_shape[0]: raise ValueError('text_mask length differs from CLIP text feature length')
            if row.get('mlm_mask') is not None and len(row['mlm_mask'])!=text_shape[0]: raise ValueError('mlm_mask length differs from CLIP text feature length')
            rows.append(row)
        except Exception as exc: errors.append(f'line {line_no}: {type(exc).__name__}: {exc}')
    if errors:
        print('\n'.join(errors[:30])); raise SystemExit(f'Aborted: {len(errors)} invalid annotations/features; no manifest written.')
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in rows)+'\n',encoding='utf-8')
    print(f'Wrote {len(rows)} rows to {out}. Video dimensions: {sum(validate_feature_array(load_array(root/ref["path"],ref["key"]),"video")[1] for ref in rows[0]["video_features"])}; text dimensions: {text_shape[1] if rows else None}')
    if not a.token_metadata: print('WARNING: token_ids/token_offsets not supplied. Add token metadata for frame-word MLM and reliable spaCy-to-CLIP subword alignment.')

if __name__=='__main__': main()
