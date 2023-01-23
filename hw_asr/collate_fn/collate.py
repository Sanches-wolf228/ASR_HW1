import logging
import torch
from typing import List

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    spec_lengths = []
    text_lengths = []
    simple_texts = []
    encoded_texts = []
    paths = []
    
    for item in dataset_items:
        spec_lengths.append(item["spectrogram"].shape[-1])
        text_lengths.append(item["text_encoded"].shape[-1])
        simple_texts.append(item["text"])
        encoded_texts.append(item["text_encoded"])
        paths.append(item["audio_path"])
       
    feature_length_dim = item["spectrogram"].shape[1]
    
    batch_spectrograms = torch.zeros(len(dataset_items), feature_length_dim, max(spec_lengths))
    batch_encoded_texts = torch.zeros(len(dataset_items), max(text_lengths))
    for i, item in enumerate(dataset_items):
        batch_encoded_texts[i, :text_lengths[i]] = item["text_encoded"]
        batch_spectrograms[i, :, :spec_lengths[i]] = item["spectrogram"]
    
    text_lengths = torch.tensor(text_lengths).int()
    batch_encoded_texts = torch.tensor(batch_encoded_texts).int()
    spec_lengths = torch.tensor(spec_lengths).int()

    return {
        "spectrogram": batch_spectrograms,
        "text_encoded": batch_encoded_texts,
        "text_encoded_length": text_lengths,
        "text": simple_texts,
        "spectrogram_length": spec_lengths,
        "audio_path" : paths
    }
