import pytest
from src.utils.colab import RunLock

def test_run_lock_ownership_and_release(tmp_path):
 path=tmp_path/'run.lock'; a=RunLock(path,who='owner-a'); b=RunLock(path,who='owner-b')
 a.acquire()
 with pytest.raises(RuntimeError): b.acquire()
 a.heartbeat(); a.release(); b.acquire(); b.release()
 assert not path.exists()
