import torchaudio
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase

class FrequencyMasking(AugmentationBase):
    def __init__(self, freq_mask_param):
        self._aug = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)

    def __call__(self, data: Tensor):
        return self._aug(data)