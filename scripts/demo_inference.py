"""Live inference demo: run a trained HM-Net / HM-Net+SE-TG checkpoint on validation queries.

Prints, for each query: the text, parser sub-events, ground-truth windows, the model's top-k
predicted windows with confidence, the IoU of the top-1 prediction, and a text timeline.

  python scripts/demo_inference.py --config <cfg.yaml> --checkpoint <best.pt> \
      --manifest <val.jsonl> --feature-root <features> --qids 2043 5266   # or --random 5
"""
import argparse, json, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.utils.config import load_config
from src.data.dataset import FeatureMomentDataset
from src.data.collate import collate_moments
from src.models.hmnet import HMNet
from src.utils.span_ops import cw_to_se

W = 60  # timeline width in characters


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0])); union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def bar(windows, duration, ch):
    cells = [' '] * W
    for s, e in windows:
        for i in range(int(s / duration * W), max(int(s / duration * W) + 1, int(round(e / duration * W)))):
            if 0 <= i < W: cells[i] = ch
    return '|' + ''.join(cells) + '|'


def build_model(cfg):
    return HMNet(video_dim=cfg.data.video_dim, text_dim=cfg.data.text_dim, hidden_dim=cfg.model.hidden_dim, heads=cfg.model.heads,
                 state_dim=cfg.model.state_dim, enhancement_layers=cfg.model.enhancement_layers, cofusion_layers=cfg.model.cofusion_layers,
                 num_queries=cfg.model.num_queries, decoder_layers=cfg.model.decoder_layers, max_seq_len=cfg.model.max_seq_len,
                 dropout=cfg.model.dropout, drop_path=cfg.model.drop_path, use_setg=cfg.model.use_setg, max_subevents=cfg.data.max_subevents,
                 use_reconstruction=cfg.model.use_reconstruction, vocab_size=cfg.data.vocab_size,
                 reconstruction_mask_fraction=cfg.reconstruction.mask_fraction, reconstruction_temperature=cfg.reconstruction.temperature,
                 reconstruction_positive_iou=cfg.reconstruction.positive_iou)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--manifest', required=True); p.add_argument('--feature-root', required=True)
    p.add_argument('--qids', nargs='*', default=[]); p.add_argument('--random', type=int, default=0)
    p.add_argument('--seed', type=int, default=0); p.add_argument('--top-k', type=int, default=5)
    a = p.parse_args()
    cfg = load_config(a.config); dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = FeatureMomentDataset(a.manifest, a.feature_root, cfg.data.max_subevents, l2_normalize=bool(cfg.data.get('l2_normalize', False)))
    index = {str(r.get('qid')): i for i, r in enumerate(ds.rows)}
    chosen = [str(q) for q in a.qids]
    missing = [q for q in chosen if q not in index]
    if missing: raise SystemExit(f'qids not in manifest: {missing}')
    if a.random: chosen += random.Random(a.seed).sample(sorted(index), a.random)
    if not chosen: raise SystemExit('give --qids or --random N')

    model = build_model(cfg).to(dev)
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(state.get('model', state)); model.eval()
    name = 'HM-Net + SE-TG' if cfg.model.use_setg else 'HM-Net (baseline)'
    print(f'Model: {name} | checkpoint {Path(a.checkpoint).name} (epoch {state.get("epoch")}) | device {dev}\n')

    for q in chosen:
        sample = ds[index[q]]; b = collate_moments([sample], cfg.data.max_subevents)
        t0 = time.time()
        with torch.no_grad():
            out = model(b['video'].to(dev), b['text'].to(dev), b['video_mask'].to(dev), b['text_mask'].to(dev),
                        b['token_spans'].to(dev), b['subevent_mask'].to(dev))
        ms = (time.time() - t0) * 1000
        d = sample['duration']; spans = cw_to_se(out['pred_spans'][0].float()).clamp(0, 1).cpu()
        prob = out['pred_logits'][0].float().softmax(-1)[:, 1].cpu(); order = prob.argsort(descending=True)[:a.top_k]
        preds = [(float(spans[i, 0]) * d, float(spans[i, 1]) * d, float(prob[i])) for i in order]
        gt = [tuple(map(float, w)) for w in ds.rows[index[q]].get('relevant_windows', [])]
        top_iou = max((iou(preds[0][:2], g) for g in gt), default=0.0)
        subs = sample['parse'].get('subevents') or sample['parse'].get('texts') or []
        print('=' * 78)
        print(f'qid {q} | video {sample["video_id"]} | duration {d:.0f}s | inference {ms:.0f} ms')
        print(f'Query: "{sample["query"]}"')
        if sample['num_subevents'] > 1:
            print(f'Sub-events (K={sample["num_subevents"]}): ' + ' | '.join(str(s.get("text", s)) if isinstance(s, dict) else str(s) for s in subs))
        print('Ground truth : ' + ', '.join(f'[{s:.0f}-{e:.0f}s]' for s, e in gt))
        print('Predictions  : ' + ', '.join(f'[{s:.1f}-{e:.1f}s] {c:.2f}' for s, e, c in preds))
        print(f'Top-1 IoU with best-matching GT: {top_iou:.2f}')
        print(f'  GT    {bar(gt, d, "#")}')
        print(f'  Top-1 {bar([preds[0][:2]], d, "=")}')
        print(f'        0s{" " * (W - 6)}{d:.0f}s')
    print('=' * 78)


if __name__ == '__main__':
    main()
