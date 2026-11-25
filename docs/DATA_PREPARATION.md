# Data preparation and manifest contract

## Primary dataset: QVHighlights
The paper uses pre-extracted SlowFast and CLIP features for QVHighlights. The project plan identifies the official [Moment-DETR repository](https://github.com/jayleicn/moment_detr) as the starting point for official annotations and feature instructions. Its QVHighlights annotations are JSONL rows containing `qid`, `query`, `duration`, `vid`, `relevant_windows`, `relevant_clip_ids`, and `saliency_scores`; the latter two fields are absent from the test split. Follow the repository's current README and licensing/terms; do not invent a direct download URL or redistribute the feature cache in this source ZIP.

Project notes commonly expect concatenated SlowFast + CLIP visual dimension 2816 and CLIP text dimension 512. These are *expected* dimensions, not verified facts about a local cache. Run the feature validator before setting the config.

## Secondary dataset: TACoS
The paper uses C3D video and GloVe text features. The adapter accepts the same manifest contract. TACoS is a stretch goal after QVHighlights works. Saliency loss must not be run when the dataset has no saliency annotations.

## Storage layout (example)
```text
nlvms/
├── features/
│   ├── qvhighlights/
│   │   ├── video/       # .npy or .npz feature arrays
│   │   ├── text/        # .npy or .npz token feature arrays
│   │   ├── train.jsonl
│   │   └── val.jsonl
│   └── tacos/
├── checkpoints/
├── logs/
└── locks/
```

Large feature files remain outside this repository/ZIP.

## Normalized JSONL manifest
One JSON object per query/video pair. Paths are relative to `--feature-root` unless absolute. Arrays are `[sequence_length, feature_dimension]`. The loader supports `.npy`, `.npz`, and keyed `.h5`/`.hdf5` datasets. `video_features` can list multiple keyed HDF5 modalities; their sequence lengths must agree and their feature dimensions are concatenated.

```json
{
  "qid": "query_0001",
  "video_id": "video_0001",
  "query": "After opening the fridge, he takes a bottle.",
  "video_feature": "video/video_0001.npy",
  "text_feature": "text/query_0001.npy",
  "duration": 42.0,
  "timestamps": [8.0, 17.5],
  "token_offsets": [[0,5],[6,13],[14,17],[18,21],[22,28],[29,32],[33,35],[36,43]],
  "token_ids": [49406, 123, 456, 789, 321, 654, 987, 49407],
  "text_mask": [1,1,1,1,1,1,1,1],
  "saliency": [[-1,-1,-1],[2,2,2],[1,0,1],[-1,-1,-1],[3,3,3]]
}
```

`token_offsets` is the exact character `[start,end)` offset for every row in `text_feature`, in the same tokenization order used by that encoder. Include special-token offsets explicitly (for example `[0,0]`) if the feature array includes special tokens. `token_ids` must use the exact tokenizer/vocabulary corresponding to the text feature rows; CLIP vocabulary size defaults to 49408 but must be verified. `text_mask` marks actual tokenizer tokens and excludes padding. `saliency` is optional and must have one entry per video timestep; entries may be three annotator ratings. Unannotated clips use `[-1,-1,-1]`. The sample above is illustrative, not a real QVHighlights record.

The adapter preserves all QVHighlights `relevant_windows` for each query. Localization uses Hungarian matching against all GT windows; SE-TG containment supervises the union of all GT windows, and segment-sentence reconstruction averages their pooled segment features. This extends the project's original single-window weak-supervision formulation to preserve the official benchmark annotations rather than silently discarding windows.

## Validation
```bash
python scripts/validate_dataset.py --manifest /path/to/train.jsonl --feature-root /path/to/features --output outputs/qvh_validation.json
```
The validator checks file existence, array rank, non-empty dimensions, NaN/Inf, feature dimensions, video lengths, timestamp validity and feature-to-manifest coverage. It does not claim official annotation completeness if the manifest itself is incomplete.

## Never do this
- Do not fill missing real features with random arrays.
- Do not infer token alignment by truncating or padding token offsets.
- Do not mix train/validation/test records during duration matching.
- Do not include feature caches or checkpoints in Git/ZIP.

## Convert official QVHighlights annotations

The converter expects the official SlowFast, CLIP-video and CLIP-text HDF5 feature stores and the official annotation JSONL. Inspect the HDF5 keys and set the paths/key convention to match your actual files. First build exact token offsets and IDs using the fast CLIP tokenizer (optional dependency):

```bash
pip install -r requirements-tokenizer.txt
python scripts/build_clip_token_metadata.py --annotations /path/highlight_train_release.jsonl --output /path/train_token_metadata.jsonl
python scripts/prepare_qvhighlights.py --annotations /path/highlight_train_release.jsonl --feature-root /path/qvhighlights --slowfast-h5 /path/qvhighlights/slowfast.hdf5 --clip-video-h5 /path/qvhighlights/clip_image.hdf5 --clip-text-h5 /path/qvhighlights/clip_text.hdf5 --token-metadata /path/train_token_metadata.jsonl --output /path/qvhighlights/train.jsonl
python scripts/validate_dataset.py --manifest /path/qvhighlights/train.jsonl --feature-root /path/qvhighlights
```

The exact HDF5 filenames and text-key convention must be checked against the downloaded feature archive. `--text-key-field` can be set to `vid` if the CLIP text store is keyed by video ID rather than query ID. The converter maps `saliency_scores` onto `relevant_clip_ids`, leaving all unannotated timesteps at -1, and retains all `relevant_windows`. The test split omits GT/saliency annotations and cannot be passed through this training-manifest converter.
