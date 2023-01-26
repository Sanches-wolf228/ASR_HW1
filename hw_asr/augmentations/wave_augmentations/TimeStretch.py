import librosa
from torch import Tensor, from_numpy

from hw_asr.augmentations.base import AugmentationBase

class TimeStretch(AugmentationBase):
    def __init__(self, rate):
        self.rate = rate

    def __call__(self, data: Tensor):
        return from_numpy(librosa.effects.time_stretch(data.cpu().numpy(), rate=self.rate))