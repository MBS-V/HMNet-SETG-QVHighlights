import torch
from torch.nn.utils.rnn import pad_sequence

def _get_gt_spans(item):
    if 'spans' in item: return item['spans']
    if 'span' in item: return item['span'].view(1,2)
    raise KeyError('Each sample needs spans or span')

def collate_moments(batch,max_subevents=4):
    videos=[x['video'] for x in batch]; texts=[x['text'] for x in batch]
    v=pad_sequence(videos,batch_first=True); q=pad_sequence(texts,batch_first=True)
    vm=torch.zeros(v.shape[:2],dtype=torch.bool); tm=torch.zeros(q.shape[:2],dtype=torch.bool)
    mm=torch.zeros(q.shape[:2],dtype=torch.bool)
    for i,x in enumerate(batch):
        vm[i,:x['video'].shape[0]]=True
        tm[i,:x['text'].shape[0]]=x.get('text_mask',torch.ones(x['text'].shape[0],dtype=torch.bool))
        mm[i,:x['text'].shape[0]]=x.get('mlm_mask',tm[i,:x['text'].shape[0]])
    spans=torch.zeros((len(batch),max_subevents,2),dtype=torch.long); sm=torch.zeros((len(batch),max_subevents),dtype=torch.bool)
    for i,x in enumerate(batch):
        k=min(x['token_spans'].shape[0],max_subevents)
        spans[i,:k]=x['token_spans'][:k]; sm[i,:k]=True
    saliency=None
    if any(x.get('saliency') is not None for x in batch):
        saliency=torch.full(v.shape[:2],-1.0,dtype=torch.float32)
        for i,x in enumerate(batch):
            if x.get('saliency') is not None:
                vals=torch.as_tensor(x['saliency'],dtype=torch.float32)
                saliency[i,:vals.shape[0]]=vals.mean(-1) if vals.ndim==2 else vals
    token_ids=None
    present=[x.get('token_ids') is not None for x in batch]
    if any(present) and not all(present): raise ValueError('token_ids must be present for every sample in a batch or omitted for all')
    if all(present):
        token_ids=torch.full(q.shape[:2],-100,dtype=torch.long)
        for i,x in enumerate(batch): token_ids[i,:x['token_ids'].shape[0]]=x['token_ids']
    return {'video':v,'text':q,'video_mask':vm,'text_mask':tm,'mlm_mask':mm,'token_ids':token_ids,'saliency':saliency,'targets':[{'spans':_get_gt_spans(x)} for x in batch],
            'gt_spans':[_get_gt_spans(x) for x in batch],'token_spans':spans,'subevent_mask':sm,
            'queries':[x['query'] for x in batch],'qids':[x['qid'] for x in batch],'video_ids':[x['video_id'] for x in batch],
            'durations':torch.tensor([x['duration'] for x in batch]),'parses':[x['parse'] for x in batch]}
