import logging
from pathlib import Path

def make_logger(name='hmnet',log_file=None,level=logging.INFO):
    logger=logging.getLogger(name); logger.setLevel(level); logger.handlers.clear()
    fmt=logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    stream=logging.StreamHandler(); stream.setFormatter(fmt); logger.addHandler(stream)
    if log_file:
        p=Path(log_file); p.parent.mkdir(parents=True,exist_ok=True)
        fh=logging.FileHandler(p,encoding='utf-8'); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger
