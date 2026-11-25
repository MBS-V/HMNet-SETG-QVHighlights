"""Dataset integrity audit for normalized manifests.

Uses exactly the same reference resolution and modality alignment as FeatureMomentDataset
(src/data/feature_loader.py), so "valid here" means "loadable in training".
Errors are grouped by category; only a few examples per category are kept.
"""
import json
from pathlib import Path
from collections import Counter, defaultdict
from .feature_loader import load_array, validate_feature_array, resolve_ref, load_video_modalities

EXAMPLES_PER_CATEGORY = 5

def _category(exc):
    msg = str(exc)
    if isinstance(exc, FileNotFoundError):
        return 'missing_text_file' if '/text' in msg.replace('\\', '/') or 'clip_text' in msg else 'missing_video_file'
    if 'length mismatch too large' in msg: return 'video_length_mismatch_too_large'
    if isinstance(exc, KeyError): return 'missing_array_key'
    if 'NaN/Inf' in msg: return 'non_finite_values'
    return type(exc).__name__

def validate_manifest(manifest, feature_root=None, max_rows=None):
    p = Path(manifest); root = Path(feature_root) if feature_root else p.parent
    rows = [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
    if max_rows: rows = rows[:max_rows]
    counts = Counter(); examples = defaultdict(list); missing_text = []
    vdim = Counter(); tdim = Counter(); lengths = []; videos = set()
    aligned_rows = 0; saliency_adjusted = 0; ok_rows = 0

    def err(cat, i, msg):
        counts[cat] += 1
        if len(examples[cat]) < EXAMPLES_PER_CATEGORY: examples[cat].append(f'row {i} (qid {rows[i].get("qid")}): {msg}')

    for i, r in enumerate(rows):
        row_ok = True
        try:
            if r.get('video_features') is not None:
                v, info = load_video_modalities(r['video_features'], root)
                if info['rows_dropped']: aligned_rows += 1
            else:
                v = load_array(*resolve_ref(r['video_feature'], root, r.get('video_key')))
            vs = validate_feature_array(v, 'video')
            vdim[vs[1]] += 1; lengths.append(vs[0]); videos.add(r.get('video_id', r.get('vid', '')))
        except Exception as e:
            err(_category(e), i, f'{type(e).__name__}: {e}'); row_ok = False; vs = None
        try:
            tpath, tkey = resolve_ref(r['text_feature'], root, r.get('text_key'))
            ts = validate_feature_array(load_array(tpath, tkey), 'text'); tdim[ts[1]] += 1
        except Exception as e:
            cat = _category(e); err(cat, i, f'{type(e).__name__}: {e}'); row_ok = False; ts = None
            if cat == 'missing_text_file':
                missing_text.append({'row': i, 'qid': r.get('qid'), 'split': r.get('split'), 'path': str(tpath)})
        dur = float(r.get('duration', 0)); windows = r.get('timestamps', r.get('relevant_windows'))
        if windows and len(windows) == 2 and all(isinstance(x, (int, float)) for x in windows): windows = [windows]
        if dur <= 0 or not windows or any(not (0 <= float(a) < float(b) <= dur + 1e-3) for a, b in windows):
            err('invalid_timestamps', i, f'duration={dur} windows={windows}'); row_ok = False
        if vs is not None and r.get('saliency') is not None and len(r['saliency']) != vs[0]:
            if abs(len(r['saliency']) - vs[0]) > 2:
                err('saliency_length_far_off', i, f'saliency {len(r["saliency"])} vs video {vs[0]}'); row_ok = False
            else:
                saliency_adjusted += 1
        if ts is not None:
            for field in ('token_ids', 'token_offsets', 'text_mask', 'mlm_mask'):
                if r.get(field) is not None and len(r[field]) != ts[0]:
                    err(f'{field}_length_vs_text_rows', i, f'{field} has {len(r[field])} entries, text feature has {ts[0]} rows'); row_ok = False
        ok_rows += int(row_ok)

    errors = [e for cat in examples for e in examples[cat]]
    return {'samples': len(rows), 'loadable_rows': ok_rows, 'videos': len(videos),
            'video_dims': dict(vdim), 'text_dims': dict(tdim),
            'video_length_min': min(lengths) if lengths else None, 'video_length_max': max(lengths) if lengths else None,
            'error_counts': dict(counts), 'error_examples': dict(examples),
            'info': {'rows_with_modality_length_aligned_by_truncation': aligned_rows,
                     'rows_with_saliency_length_adjusted_by_one_clip': saliency_adjusted},
            'missing_text_files': missing_text,
            'errors': errors, 'valid': not counts}
