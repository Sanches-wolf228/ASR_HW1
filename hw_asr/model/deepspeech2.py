import torch
import torch.nn.functional as F
from torch import nn

from hw_asr.base import BaseModel

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=256, bidirectional=True):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, x, lengths): # batch, time, dim
        b, t, d = x.shape
        x = x.reshape(-1, d) # batch * time, dim
        x = self.bn(x)
        x = x.view(b, -1, d) # batch, time, dim
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False, batch_first=True)
        x, _ = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        s = x.shape[2] // 2
        x = x[:, :, :s] + x[:, :, s:]
        return x

class DeepSpeech2(BaseModel):
    def __init__(self, n_feats, n_class, num_rnn_layers=5, rnn_hidden_size=256, **batch):
        super().__init__(n_feats, n_class, **batch)
        self.cnn = nn.Conv2d(1, 32, kernel_size=(41, 21), stride=2, padding=(20, 10))
        self.rnns = nn.ModuleList()
        rnn_input_size = n_feats * 32 // 2

        for i in range(num_rnn_layers):
            self.rnns.append(RNN(
                input_size=rnn_input_size if i == 0 else rnn_hidden_size,
                hidden_size=rnn_hidden_size
            ))

        self.fc = nn.Sequential(
            nn.LayerNorm(rnn_hidden_size),
            nn.Linear(rnn_hidden_size, n_class)
        )

    def forward(self, spectrogram, spectrogram_length, **batch):
        b, f, t = spectrogram.shape # batch, n_feats, time
        lengths = self.transform_input_lengths(spectrogram_length)
        x = self.cnn(spectrogram.view(b, 1, f, t))
        x = x.view(b, -1, x.shape[3]).transpose(1, 2) # batch, time, dim
        
        for rnn in self.rnns:
            x = rnn(x, lengths)

        x = x.reshape(-1, x.shape[2]) # batch * time, dim
        x = self.fc(x) # batch * time, dim
        x = x.view(b, -1, x.shape[1]) # batch, time, dim
        return {"logits": x} # batch, time, n_class

    def transform_input_lengths(self, input_lengths):
        return ((input_lengths + 1) // 2 + 1) // 2