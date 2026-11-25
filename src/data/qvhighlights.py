"""QVHighlights adapter over the validated normalized feature-manifest interface."""
from .dataset import FeatureMomentDataset
class QVHighlightsDataset(FeatureMomentDataset):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
