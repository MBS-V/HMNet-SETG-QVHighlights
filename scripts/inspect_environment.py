import sys,platform,importlib.util
print('Python:',sys.version); print('Platform:',platform.platform())
for p in ['torch','numpy','scipy','omegaconf','spacy','mamba_ssm']:
 print(p, bool(importlib.util.find_spec(p)))
try:
 import torch
 print('PyTorch:',torch.__version__,'CUDA runtime:',torch.version.cuda,'CUDA available:',torch.cuda.is_available())
 if torch.cuda.is_available(): print('GPU:',torch.cuda.get_device_name(0))
except Exception as e: print('Torch inspection failed:',e)
