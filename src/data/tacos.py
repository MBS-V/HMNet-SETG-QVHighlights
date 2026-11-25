"""TACoS adapter over the normalized manifest interface; saliency is not assumed."""
from .dataset import FeatureMomentDataset
class TACoSDataset(FeatureMomentDataset):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
