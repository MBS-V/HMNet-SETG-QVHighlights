"""Regression tests for the Nov-2026 real-data audit fixes."""
import json, subprocess, sys
from pathlib import Path
import numpy as np, pytest, torch
from src.data.dataset import FeatureMomentDataset
from src.data.feature_validation import validate_manifest
from src.data.feature_loader import resolve_ref, load_video_modalities
from src.models.mamba_block import BidirectionalMamba, masked_flip
from src.losses.reconstruction import MESMReconstruction

ROOT = Path(__file__).resolve().parents[1]

def _write(tmp, rows):
    m = tmp / 'm.jsonl'; m.write_text('\n'.join(json.dumps(r) for r in rows) + '\n'); return m

def _row(tmp, qid, Lsf=76, Lclip=75, text_rows=4, dict_text=True, sal_len=75):
    (tmp / 'video').mkdir(exist_ok=True); (tmp / 'text').mkdir(exist_ok=True)
    np.savez(tmp / f'video/sf{qid}.npz', features=np.random.rand(Lsf, 6).astype(np.float32))
    np.savez(tmp / f'video/cl{qid}.npz', features=np.random.rand(Lclip, 2).astype(np.float32))
    np.savez(tmp / f'text/qid{qid}.npz', last_hidden_state=np.random.rand(text_rows, 3).astype(np.float32), pooler_output=np.random.rand(3))
    tf = {'path': f'text/qid{qid}.npz', 'key': 'last_hidden_state'} if dict_text else f'text/qid{qid}.npz'
    r = {'qid': qid, 'video_id': f'v{qid}', 'query': 'a man walks', 'duration': 150.0,
         'video_features': [{'path': f'video/sf{qid}.npz', 'key': 'features'}, {'path': f'video/cl{qid}.npz', 'key': 'features'}],
         'text_feature': tf, 'relevant_windows': [[10, 20], [30, 40]], 'token_ids': [49406, 1, 2, 49407],
         'token_offsets': [[0, 0], [0, 1], [2, 5], [0, 0]], 'text_mask': [1, 1, 1, 1],
         'saliency': [[-1, -1, -1]] * sal_len}
    if not dict_text: r['text_key'] = 'last_hidden_state'
    return r

def test_resolve_ref_both_schemas(tmp_path):
    assert resolve_ref({'path': 'a/b.npz', 'key': 'k'}, tmp_path) == (tmp_path / 'a/b.npz', 'k')
    assert resolve_ref('a/b.npz', tmp_path, 'k') == (tmp_path / 'a/b.npz', 'k')
    with pytest.raises(TypeError):
        resolve_ref(5, tmp_path)

def test_dict_and_legacy_text_refs_load_and_validate(tmp_path):
    rows = [_row(tmp_path, 1), _row(tmp_path, 2, dict_text=False, Lsf=75)]
    m = _write(tmp_path, rows)
    rep = validate_manifest(m, tmp_path)
    assert rep['valid'], rep['error_counts']
    assert rep['info']['rows_with_modality_length_aligned_by_truncation'] == 1
    ds = FeatureMomentDataset(m, tmp_path)
    for i in range(2):
        it = ds[i]
        assert it['video'].shape == (75, 8) and it['text'].shape == (4, 3) and it['spans'].shape == (2, 2)

def test_official_style_truncation_and_large_mismatch(tmp_path):
    _row(tmp_path, 3, Lsf=76, Lclip=75)
    v, info = load_video_modalities([{'path': 'video/sf3.npz', 'key': 'features'}, {'path': 'video/cl3.npz', 'key': 'features'}], tmp_path)
    assert v.shape == (75, 8) and info == {'modality_lengths': [76, 75], 'rows_dropped': 1}
    raw = np.load(tmp_path / 'video/sf3.npz')['features']
    assert np.allclose(v[:, :6], raw[:75])            # leading rows kept, only the trailing row dropped
    _row(tmp_path, 4, Lsf=80, Lclip=75)
    with pytest.raises(ValueError, match='too large'):
        load_video_modalities([{'path': 'video/sf4.npz'}, {'path': 'video/cl4.npz'}], tmp_path)

def test_missing_text_reported_as_category(tmp_path):
    r = _row(tmp_path, 5); (tmp_path / 'text/qid5.npz').unlink()
    rep = validate_manifest(_write(tmp_path, [r]), tmp_path)
    assert rep['error_counts'] == {'missing_text_file': 1} and rep['missing_text_files'][0]['qid'] == 5

def test_saliency_off_by_one_is_aligned(tmp_path):
    r = _row(tmp_path, 6, Lsf=74, Lclip=74, sal_len=75)
    it = FeatureMomentDataset(_write(tmp_path, [r]), tmp_path)[0]
    assert len(it['saliency']) == it['video'].shape[0] == 74

def test_masked_flip_and_padding_invariance():
    x = torch.arange(6.).view(1, 6, 1); m = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    assert masked_flip(x, m)[0, :, 0].tolist() == [3, 2, 1, 0, 4, 5]
    torch.manual_seed(0); bm = BidirectionalMamba(8, 4, prefer_cuda=False).eval()
    a = torch.randn(1, 5, 8)
    with torch.no_grad():
        y1 = bm(a)
        y2 = bm(torch.cat([a, torch.randn(1, 3, 8)], 1), padding_mask=torch.tensor([[1] * 5 + [0] * 3], dtype=torch.bool))[:, :5]
    assert torch.allclose(y1, y2, atol=1e-5)

def test_segment_sentence_loss_is_bounded_like_mesm():
    torch.manual_seed(0); rec = MESMReconstruction(16, vocab_size=10, heads=4, temperature=0.5)
    v = torch.randn(4, 9, 16) * 50; q = torch.randn(4, 7, 16) * 50        # large magnitudes
    l = rec.segment_sentence_loss(v, q, torch.ones(4, 9, dtype=torch.bool), torch.ones(4, 7, dtype=torch.bool),
                                  [torch.tensor([[0.1, 0.5]])] * 4, ['a', 'b', 'c', 'd'])
    # cosine/0.5 logits lie in [-2, 2] -> loss <= log(4) + 4
    assert torch.isfinite(l) and 0 <= l.item() <= np.log(4) + 4

def test_audit_text_features_script(tmp_path):
    r = _row(tmp_path, 7); m = _write(tmp_path, [r, _row(tmp_path, 8)])
    (tmp_path / 'text/qid8.npz').rename(tmp_path / 'qid8.npz')
    bak = tmp_path / 'bak'; bak.mkdir(); (tmp_path / 'qid8.npz').rename(bak / 'qid8.npz')
    py = sys.executable; s = str(ROOT / 'scripts/audit_text_features.py')
    r1 = subprocess.run([py, s, '--feature-root', str(tmp_path), '--manifests', str(m), '--output', str(tmp_path / 'a.json')], capture_output=True, text=True)
    assert r1.returncode == 3 and 'missing: 1' in r1.stdout
    r2 = subprocess.run([py, s, '--feature-root', str(tmp_path), '--manifests', str(m), '--search-dir', str(bak), '--copy-missing-from',
                         '--output', str(tmp_path / 'b.json')], capture_output=True, text=True)
    assert r2.returncode == 0 and (tmp_path / 'text/qid8.npz').exists()
