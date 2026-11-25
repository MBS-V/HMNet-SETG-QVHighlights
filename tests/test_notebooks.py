from pathlib import Path
import nbformat

def test_all_notebooks_are_valid():
 files=sorted(Path('notebooks').glob('*.ipynb'))
 names={p.name for p in files}
 assert {f'0{i}' for i in range(5)} <= {n[:2] for n in names} and len(files)>=5  # 00-04 + recovery notebook 05
 for p in files: nbformat.validate(nbformat.read(p,as_version=4))
