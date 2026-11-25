"""Audit / restore CLIP *query* text features referenced by the manifests.

What it does (read-only unless --copy-missing-from is given):
  1. lists every text file the manifests reference and checks it exists under --feature-root;
  2. checks a sample of existing files: key present, 2-D, dim 512, rows == len(token_ids);
  3. optionally looks for the missing files in one or more backup directories (e.g. the
     Google Drive copy, or a fresh extraction of moment_detr_features.tar.gz) and copies ONLY
     the missing ones (never overwrites an existing file), then re-checks;
  4. optionally writes "<split>_available.jsonl" manifests containing only fully loadable rows
     (for smoke/debug runs ONLY - final results must use the complete manifests).

It never looks in clip_sub_features: those are ASR-subtitle features for Moment-DETR's
weakly supervised pretraining (scripts/pretrain.sh, data/subs_train.jsonl), not query features.

Example (Colab):
  python scripts/audit_text_features.py --feature-root /content/qvhighlights \
      --manifests /content/qvhighlights/train.jsonl /content/qvhighlights/val.jsonl \
      --search-dir /content/drive/MyDrive/nlvms/features/qvhighlights/text/clip_text_features
"""
import argparse, json, random, shutil, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.feature_loader import load_array, resolve_ref  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--feature-root', required=True)
    ap.add_argument('--manifests', nargs='+', required=True)
    ap.add_argument('--search-dir', nargs='*', default=[], help='directories that may contain qid*.npz copies')
    ap.add_argument('--copy-missing-from', action='store_true', help='copy missing files found in --search-dir (no overwrite)')
    ap.add_argument('--write-available', action='store_true', help='write <manifest>_available.jsonl (debug only)')
    ap.add_argument('--check-sample', type=int, default=300)
    ap.add_argument('--output', default='outputs/text_feature_audit.json')
    a = ap.parse_args()
    root = Path(a.feature_root)

    refs = []  # (manifest, row_index, qid, split, path, key, n_tokens)
    for m in a.manifests:
        for i, line in enumerate(Path(m).read_text(encoding='utf-8').splitlines()):
            if not line.strip():
                continue
            r = json.loads(line)
            path, key = resolve_ref(r['text_feature'], root, r.get('text_key'))
            refs.append((m, i, r.get('qid'), r.get('split'), path, key, len(r['token_ids']) if r.get('token_ids') else None))
    missing = [x for x in refs if not x[4].exists()]
    text_dirs = {x[4].parent for x in refs}
    on_disk = sum(len(list(d.glob('*.npz'))) for d in text_dirs if d.exists())
    by_split = Counter(x[3] for x in missing)
    print(f'referenced text files: {len(refs)} | missing: {len(missing)} {dict(by_split)} | .npz files in referenced dir(s): {on_disk}')

    # Content check on a sample of existing files
    present = [x for x in refs if x[4].exists()]
    random.seed(0)
    sample = random.sample(present, min(a.check_sample, len(present)))
    shape_issues = Counter(); dims = Counter(); row_rel = Counter()
    for _, _, qid, _, path, key, ntok in sample:
        try:
            arr = load_array(path, key)
            if arr.ndim != 2:
                shape_issues['not_2d'] += 1; continue
            dims[int(arr.shape[1])] += 1
            if ntok is not None:
                row_rel['rows==len(token_ids)' if arr.shape[0] == ntok else f'rows-len(token_ids)={arr.shape[0]-ntok}'] += 1
        except Exception as e:  # noqa: BLE001
            shape_issues[type(e).__name__] += 1
    print(f'sample of {len(sample)} existing files: dims {dict(dims)} | rows vs token_ids {dict(row_rel)} | problems {dict(shape_issues)}')

    # Look for missing files in backup locations
    found = {}
    for d in a.search_dir:
        d = Path(d)
        if not d.exists():
            print(f'search dir does not exist: {d}'); continue
        names = {p.name: p for p in d.rglob('qid*.npz') if '_subs' not in p.name}
        hit = 0
        for x in missing:
            if x[4].name in names and x[4] not in found:
                found[x[4]] = names[x[4].name]; hit += 1
        print(f'{d}: {len(names)} qid*.npz files, contains {hit} of the missing files')
    copied = 0
    if a.copy_missing_from and found:
        for dst, src in found.items():
            if dst.exists():
                continue  # never overwrite
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.parent / (dst.name + '.part')
            shutil.copy2(src, tmp)
            try:  # verify the copy is a readable npz before it becomes visible under the real name
                import numpy as np
                with np.load(tmp) as z:
                    if not z.files: raise ValueError('empty npz')
            except Exception as e:  # noqa: BLE001
                tmp.unlink(missing_ok=True); print(f'skipping unreadable copy of {src}: {e}'); continue
            tmp.rename(dst); copied += 1
        print(f'copied {copied} missing files (existing files untouched)')
    still_missing = [x for x in refs if not x[4].exists()]

    if a.write_available:
        for m in a.manifests:
            bad = {x[1] for x in still_missing if x[0] == m}
            lines = [l for l in Path(m).read_text(encoding='utf-8').splitlines() if l.strip()]
            out = Path(m).with_name(Path(m).stem + '_available.jsonl')
            out.write_text('\n'.join(l for i, l in enumerate(lines) if i not in bad) + '\n', encoding='utf-8')
            print(f'wrote {out} ({len(lines) - len(bad)} of {len(lines)} rows) - DEBUG ONLY, not for final results')

    rep = {'referenced': len(refs), 'missing_before': len(missing), 'missing_by_split': dict(by_split),
           'npz_on_disk': on_disk, 'found_in_search_dirs': len(found), 'copied': copied,
           'missing_after': len(still_missing), 'sample_dims': dict(dims), 'sample_rows_vs_tokens': dict(row_rel),
           'sample_problems': dict(shape_issues),
           'missing_files': [{'qid': x[2], 'split': x[3], 'path': str(x[4])} for x in still_missing]}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(rep, indent=2))
    print(f'STILL MISSING: {len(still_missing)}  (full list in {a.output})')
    return 0 if not still_missing else 3


if __name__ == '__main__':
    sys.exit(main())
