"""Manifest-driven feature dataset. Each JSONL row points to one video/text feature pair."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .feature_loader import load_array,validate_feature_array,resolve_ref,load_video_modalities,l2_normalize
from ..setg.parser import SubEventParser
from ..setg.token_alignment import regex_token_offsets,char_spans_to_token_spans

class FeatureMomentDataset(Dataset):
    def __init__(self,manifest,feature_root=None,max_subevents=4,validate=True,l2_normalize=False):
        self.path=Path(manifest); self.root=Path(feature_root) if feature_root else self.path.parent
        self.rows=[json.loads(x) for x in self.path.read_text(encoding='utf-8').splitlines() if x.strip()]
        self.parser=SubEventParser(max_subevents=max_subevents)
        self.validate=validate; self.l2=l2_normalize
    def __len__(self): return len(self.rows)
    def _resolve(self,p):
        p=Path(p); return p if p.is_absolute() else self.root/p
    def __getitem__(self,i):
        r=self.rows[i]
        video_refs=r.get('video_features')
        if video_refs is not None:
            # Same alignment as the official Moment-DETR loader: truncate to the shortest modality.
            v,_=load_video_modalities(video_refs,self.root,normalize=self.l2)
        else:
            v=load_array(*resolve_ref(r['video_feature'],self.root,r.get('video_key')))
        q=load_array(*resolve_ref(r['text_feature'],self.root,r.get('text_key')))
        if self.l2: q=l2_normalize(q)
        if self.validate: validate_feature_array(v,'video'); validate_feature_array(q,'text')
        query=r.get('query',''); parsed=self.parser.parse(query)
        offsets=r.get('token_offsets')
        if offsets is None:
            # This fallback is valid only when precomputed text rows follow regex tokenization.
            offsets=regex_token_offsets(query)
            if len(offsets)!=q.shape[0]:
                raise ValueError(f"{r.get('qid',i)}: text feature length={q.shape[0]} but regex tokens={len(offsets)}; include exact token_offsets in manifest")
        token_spans=char_spans_to_token_spans(parsed.char_spans,offsets)
        token_ids=r.get('token_ids')
        text_mask=r.get('text_mask',[1]*q.shape[0])
        mlm_mask=r.get('mlm_mask',text_mask)
        if len(mlm_mask)!=q.shape[0]: raise ValueError(f"{r.get('qid',i)}: mlm_mask length {len(mlm_mask)} != text feature length {q.shape[0]}")
        if len(text_mask)!=q.shape[0]: raise ValueError(f"{r.get('qid',i)}: text_mask length {len(text_mask)} != text feature length {q.shape[0]}")
        if token_ids is not None and len(token_ids)!=q.shape[0]:
            raise ValueError(f"{r.get('qid',i)}: token_ids length {len(token_ids)} != text feature length {q.shape[0]}")
        k=len(token_spans)
        raw_spans=r.get('timestamps',r.get('relevant_windows'))
        duration=float(r.get('duration',0))
        if raw_spans is None: raise ValueError(f"Manifest row {i} missing timestamps/relevant_windows")
        if len(raw_spans)==2 and all(isinstance(x,(int,float)) for x in raw_spans): raw_spans=[raw_spans]
        if duration<=0: raise ValueError(f"Manifest row {i} needs positive video duration")
        normalized=[]
        for window in raw_spans:
            start,end=map(float,window)
            if not (0<=start<end<=duration+1e-3): raise ValueError(f"Invalid timestamps {window} for duration {duration}")
            normalized.append([start/duration,end/duration])
        if not normalized: raise ValueError(f"Manifest row {i} has no valid ground-truth windows")
        saliency=r.get('saliency')
        if saliency is not None and len(saliency)!=v.shape[0]:
            # Saliency has one row per 2-s clip of the annotation (round(duration/2)); features may have
            # +/-1 row. Like Moment-DETR (which ignores clip ids >= ctx_l) we keep clips < L and pad with -1.
            if abs(len(saliency)-v.shape[0])>2:
                raise ValueError(f"{r.get('qid',i)}: saliency length {len(saliency)} far from video feature length {v.shape[0]}")
            width=len(saliency[0]) if len(saliency) else 3
            saliency=list(saliency[:v.shape[0]])+[[-1]*width]*max(0,v.shape[0]-len(saliency))
        return {'video':torch.as_tensor(np.asarray(v).copy(),dtype=torch.float32),'text':torch.as_tensor(np.asarray(q).copy(),dtype=torch.float32),
                'query':query,'qid':str(r.get('qid',i)),'video_id':str(r.get('video_id',r.get('vid',''))),'duration':duration,
                'spans':torch.tensor(normalized,dtype=torch.float32),'text_mask':torch.tensor(text_mask,dtype=torch.bool),'mlm_mask':torch.tensor(mlm_mask,dtype=torch.bool),
                'token_spans':torch.tensor(token_spans,dtype=torch.long),'token_ids':torch.tensor(token_ids,dtype=torch.long) if token_ids is not None else None,'num_subevents':k,'parse':parsed.to_dict(),
                'saliency':saliency}
