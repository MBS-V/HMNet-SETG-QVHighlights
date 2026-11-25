from src.utils.config import load_config

def test_dataset_config_merges_defaults():
 c=load_config('configs/qvhighlights.yaml')
 assert c.model.hidden_dim==256 and c.data.video_dim==2816 and c.model.use_setg is False

def test_setg_config_merges_defaults():
 c=load_config('configs/hmnet_setg.yaml')
 assert c.model.use_setg is True and c.loss.weight_sub==0.1 and c.paths.checkpoints


def test_explicit_dataset_setg_configs():
 q=load_config('configs/qvhighlights_setg.yaml'); t=load_config('configs/tacos_setg.yaml')
 assert q.data.dataset=='qvhighlights' and q.model.use_setg
 assert t.data.dataset=='tacos' and t.model.use_setg and t.loss.weight_sal==0.0
