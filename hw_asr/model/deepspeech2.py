import torch
import torch.nn.functional as F
from torch import nn

from hw_asr.base import BaseModel

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential( 
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        )
        self.conv2 = nn.Sequential( 
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        )

    def _pad_sequence(self, x, lengths):
        # batch, conv_filters, dim, time
        mask = torch.zeros_like(x, dtype=torch.bool)
        time = x.shape[-1]
        for i, length in enumerate(lengths):
            d_time = time - length
            if d_time > 0:
                mask[i].narrow(dim = -1, start = length, length = d_time).fill_(1)
        x.masked_fill(mask, 0)
        return x

    def forward(self, x, lengths):
        x = self._pad_sequence(self.conv1(x), lengths)
        x = self._pad_sequence(self.conv2(x), lengths)
        return x

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
        x = x.view(b, t, d) # batch, time, dim
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False, batch_first = True)
        x, _ = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        s = x.shape[2] // 2
        x = x[:, :, :s] + x[:, :, s:]
        return x

class DeepSpeech2(BaseModel):
    def __init__(self, n_feats, n_class, num_rnn_layers=3, rnn_hidden_size=256, **batch):
        super().__init__(n_feats, n_class, **batch)
        self.cnn = CNN()
        self.rnns = nn.ModuleList()
        rnn_input_size = n_feats // 4 * 32

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
        b, f, t = spectrogram.shape # batch, feats, time
        lengths = self.transform_input_lengths(spectrogram_length)
        x = self.cnn(spectrogram.view(b, 1, f, t), lengths)
        x = x.view(b, -1, x.shape[3]).transpose(1, 2) # batch * time, dim
        
        for rnn in self.rnns:
            x = rnn(x, lengths)

        x = x.reshape(-1, x.shape[2]) # batch * time, dim
        x = self.fc(x) # batch * time, dim
        x = x.view(b, -1, x.shape[1]) # batch, time, dim
        return {"logits": x} # batch, time, n_class

    def transform_input_lengths(self, input_lengths):
        return torch.div(input_lengths, 2, rounding_mode='trunc')
