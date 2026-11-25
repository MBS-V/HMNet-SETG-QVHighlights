# Google Colab workflow (T4 target)

## Shared Drive setup (one-time)
1. One teammate creates `MyDrive/nlvms/` with `features/`, `checkpoints/`, `logs/`, and `locks/` subfolders.
2. Share the folder with the other teammates as Editor.
3. Each teammate adds a shortcut from Shared with me to their own My Drive.
4. Place the official precomputed feature cache in `nlvms/features/qvhighlights/` once.
5. Each experiment uses its own checkpoint directory. The Drive lock is advisory only; Drive sync is not a strongly consistent distributed lock service.

## Notebook order
- `00_colab_setup.ipynb`: inspect Python/PyTorch/CUDA/GPU, mount Drive, install only non-Torch dependencies, resolve project root/config.
- `01_colab_data_validation.ipynb`: verify manifest and real feature arrays; inspect dimensions and errors.
- `02_colab_smoke_test.ipynb`: run CPU/GPU model and checkpoint smoke tests on synthetic data.
- `03_colab_training.ipynb`: run the configured training command; checkpoints resume by default.
- `04_colab_evaluation.ipynb`: evaluate a supplied checkpoint and generate metrics/prediction files.

Notebook 00 attempts to install `en_core_web_sm` for dependency parsing; if network access is unavailable, the parser falls back to connective rules. Install `requirements-tokenizer.txt` separately when generating exact CLIP token offsets/IDs.

The notebooks are executable code cells, not screenshots. They assume you have uploaded/extracted this ZIP into Colab or mounted the project directory in Drive. GitHub is not required for initial execution.

## T4 specifics
- T4 is a Turing GPU: use FP16 with GradScaler or FP32; do not assume native BF16.
- PyTorch is supplied by the Colab runtime. Do not reinstall/downgrade it just to install Mamba.
- The pure-PyTorch fallback is used when the fused `mamba_ssm` extension is absent. It is much slower; the QVHighlights sequence lengths make smoke tests reasonable, not full-scale training guaranteed.
- Attempt the optional official `mamba-ssm` installation only in a separate session after inspecting exact PyTorch/CUDA/Python/compiler compatibility. Restart runtime after installation and verify import before model construction.

## After runtime disconnect
1. Reconnect and rerun notebook 00 (GPU check, Drive mount, config).
2. Rerun notebook 01 if the feature cache or manifest changed; otherwise confirm the validation report exists.
3. Run notebook 02 smoke test after environment/package changes.
4. In notebook 03, use the same `--run-name` and do not pass `--fresh`; the trainer loads the newest valid checkpoint.
5. Verify the printed resume epoch and `checkpoints/<run-name>/latest.json`.
6. Notebook 04 evaluates the selected `best.pt` only if that file exists and is valid.

Do not train two accounts into the same run. Use separate run names or coordinate lock ownership.
