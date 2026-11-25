import json, numpy as np, h5py
from pathlib import Path
from src.data.dataset import FeatureMomentDataset
from src.data.collate import collate_moments
from src.data.feature_validation import validate_manifest

def test_hdf5_modalities_multi_gt_and_saliency(tmp_path):
    video_a=tmp_path/'slowfast.h5'; video_b=tmp_path/'clip.h5'; text_p=tmp_path/'text.h5'
    with h5py.File(video_a,'w') as f: f.create_dataset('vid1',data=np.ones((5,3),dtype='float32'))
    with h5py.File(video_b,'w') as f: f.create_dataset('vid1',data=np.ones((5,2),dtype='float32'))
    with h5py.File(text_p,'w') as f: f.create_dataset('q1',data=np.ones((2,4),dtype='float32'))
    row={'qid':'q1','vid':'vid1','query':'go now','duration':10,'relevant_windows':[[1,3],[4,5]],
         'video_features':[{'path':'slowfast.h5','key':'vid1'},{'path':'clip.h5','key':'vid1'}],
         'text_feature':'text.h5','text_key':'q1','token_offsets':[[0,2],[3,6]],'token_ids':[1,2],
         'saliency':[[-1,-1,-1],[4,3,4],[2,2,2],[-1,-1,-1],[0,0,0]]}
    manifest=tmp_path/'train.jsonl'; manifest.write_text(json.dumps(row))
    report=validate_manifest(manifest,tmp_path); assert report['valid'] and report['video_dims']=={5:1}
    ds=FeatureMomentDataset(manifest,tmp_path,use_spacy=False) if False else FeatureMomentDataset(manifest,tmp_path)
    item=ds[0]; assert item['video'].shape==(5,5) and item['spans'].shape==(2,2)
    batch=collate_moments([item]); assert batch['gt_spans'][0].shape==(2,2)
    assert batch['token_ids'].shape==(1,2) and batch['saliency'].shape==(1,5)
