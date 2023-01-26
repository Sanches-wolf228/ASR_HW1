import torchaudio
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase

class PitchShift(AugmentationBase):
    def __init__(self, sample_rate, n_steps, *args, **kwargs):
        self._aug = torchaudio.transforms.PitchShift(sample_rate, n_steps, *args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)