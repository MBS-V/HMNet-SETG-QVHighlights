from pathlib import Path
import numpy as np

# Largest allowed difference (rows) between video modalities of one video.
# The official Moment-DETR loader (start_end_dataset.py::_get_video_feat_by_vid) truncates
# every modality to the shortest one with the comment "some features are slightly longer
# than the others". Real QVHighlights shows differences of exactly 1 row (SlowFast one
# trailing clip longer than CLIP). Larger differences indicate a wrong file and are errors.
MAX_MODALITY_LENGTH_DIFF = 2

def load_array(path,key=None):
    path=Path(path)
    if not path.exists(): raise FileNotFoundError(f"Feature file not found: {path}")
    if path.suffix=='.npy': return np.load(path,mmap_mode='r')
    if path.suffix in {'.h5','.hdf5'}:
        import h5py
        if key is None: raise ValueError(f'HDF5 feature file {path} requires an explicit dataset key')
        with h5py.File(path,'r') as h:
            if key not in h: raise KeyError(f'Key {key!r} absent from {path}')
            return np.asarray(h[key])
    if path.suffix=='.npz':
        z=np.load(path,mmap_mode='r')
        if key is not None:
            if key not in z: raise KeyError(f"Key {key!r} absent from {path}; available={list(z.files)}")
            return z[key]
        if len(z.files)!=1: raise ValueError(f"NPZ {path} has multiple arrays; specify feature key")
        return z[z.files[0]]
    raise ValueError(f"Unsupported feature format {path.suffix}; use .npy or .npz")

def validate_feature_array(array,name='feature'):
    a=np.asarray(array)
    if a.ndim!=2: raise ValueError(f"{name} must be [sequence,dim], got {a.shape}")
    if not np.isfinite(a).all(): raise ValueError(f"{name} contains NaN/Inf")
    if a.shape[0]<1 or a.shape[1]<1: raise ValueError(f"{name} has empty dimension: {a.shape}")
    return tuple(a.shape)

def resolve_ref(ref,root,legacy_key=None):
    """Return (absolute_path, key) for a manifest feature reference.

    Supports both manifest schemas that exist for this project:
      * dict  {"path": "text/clip_text_features/qid9769.npz", "key": "last_hidden_state"}
      * str   "clip_text.h5"  with the key in a separate row field (legacy ``text_key``/``video_key``)
    Relative paths are resolved against ``root``.
    """
    if isinstance(ref,dict):
        if 'path' not in ref: raise ValueError(f"feature reference dict has no 'path': {ref}")
        path,key=ref['path'],ref.get('key',legacy_key)
    elif isinstance(ref,(str,Path)):
        path,key=ref,legacy_key
    else:
        raise TypeError(f"feature reference must be a str or a {{'path','key'}} dict, got {type(ref).__name__}")
    p=Path(str(path))
    return (p if p.is_absolute() else Path(root)/p),key

def l2_normalize(a,eps=1e-5):
    a=np.asarray(a,dtype=np.float32); return a/(np.linalg.norm(a,axis=-1,keepdims=True)+eps)

def load_video_modalities(refs,root,max_diff=MAX_MODALITY_LENGTH_DIFF,normalize=False):
    """Load and concatenate video modalities, aligning lengths exactly like the official loader.

    Returns (array [L, sum(dims)], info) where info records original lengths and rows dropped.
    Truncation keeps rows 0..L-1 of every modality (drops trailing rows only), which is exactly
    what the official Moment-DETR loader does; QVHighlights annotations index the same 2-second
    clips from t=0, and saliency/relevant_clip_ids never point past round(duration/2).
    """
    arrays=[]
    for ref in refs:
        path,key=resolve_ref(ref,root)
        arr=load_array(path,key); arrays.append(l2_normalize(arr) if normalize else arr)
    lengths=[int(a.shape[0]) for a in arrays]
    diff=max(lengths)-min(lengths)
    if diff>max_diff:
        raise ValueError(f"video modality length mismatch too large ({lengths}, dims {[a.shape[1] for a in arrays]}); "
                         f"official alignment only tolerates small trailing differences")
    L=min(lengths)
    v=np.concatenate([np.asarray(a[:L]) for a in arrays],axis=-1)
    return v,{'modality_lengths':lengths,'rows_dropped':diff}
