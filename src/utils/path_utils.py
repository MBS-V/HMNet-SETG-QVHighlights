from pathlib import Path

def resolve_path(path,base=None):
    p=Path(path).expanduser()
    if not p.is_absolute() and base is not None: p=Path(base)/p
    return p.resolve()
