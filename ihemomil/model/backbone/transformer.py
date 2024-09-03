import torch
from torch import nn
import math

class TransformerFeatureExtractor(nn.Module):
    def __init__(self, n_in_channels: int, n_out_channels: int, nhead: int, nhid: int, nlayers: int):
        super().__init__()
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(n_in_channels)
        encoder_layers = nn.TransformerEncoderLayer(n_in_channels, nhead, nhid)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Linear(n_in_channels, n_in_channels * n_out_channels)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        if self.src_mask is None or self.src_mask.size(0) != len(x):
            device = x.device
            mask = self._generate_square_subsequent_mask(len(x)).to(device)
            self.src_mask = mask

        x = x * math.sqrt(self.n_in_channels)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x, self.src_mask)
        output = self.decoder(output)
        output = output.view(-1, self.n_out_channels, self.n_in_channels)  # Reshape the output
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)