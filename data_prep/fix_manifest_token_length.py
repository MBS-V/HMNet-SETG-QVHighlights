"""Cut token_ids / token_offsets / text_mask to the length of the official CLIP text feature (32 tokens max).

8 QVHighlights queries (3 train, 4 val, 1 test) have more than 32 CLIP tokens, but the official
clip_text_features keep only the first 32 rows. Writes new manifests to <manifest_dir>/fixed/.
Usage: python data_prep/fix_manifest_token_length.py <manifest_dir> <feature_root>
"""
import json, sys
from pathlib import Path
import numpy as np

mdir, froot = Path(sys.argv[1]), Path(sys.argv[2]); (mdir / 'fixed').mkdir(exist_ok=True); total = 0
for split in ['train', 'val', 'test']:
    k = 0
    with open(mdir / 'fixed' / f'{split}.jsonl', 'w') as out:
        for line in open(mdir / f'{split}.jsonl'):
            r = json.loads(line); L = np.load(froot / r['text_feature']['path'])['last_hidden_state'].shape[0]
            if len(r['token_ids']) > L:
                for key in ('token_ids', 'token_offsets', 'text_mask'): r[key] = r[key][:L]
                r['text_truncated_to'] = L; k += 1
            out.write(json.dumps(r) + '\n')
    print(split, 'rows truncated:', k); total += k
print('total truncated', total)
