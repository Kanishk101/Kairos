import torch
import torch.nn as nn
import math

def _num_groups(channels):
    for g in [8, 6, 4, 3, 2]:
        if channels % g == 0:
            return g
    return 1

class ConvBlock1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dropout=0.15):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.norm = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(self.norm(self.conv(x))))

class ResidualConvBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dropout=0.15):
        super().__init__()
        self.block1 = ConvBlock1d(channels, channels, kernel_size, dropout)
        self.block2 = ConvBlock1d(channels, channels, kernel_size, dropout)

    def forward(self, x):
        return x + self.block2(self.block1(x))

class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ProposedModel(nn.Module):
    def __init__(self, C=11, T=60, width=48, dropout=0.15, n_classes=3):
        super().__init__()
        self.stem = nn.Conv1d(C, width, kernel_size=1)
        self.conv = nn.Sequential(
            ConvBlock1d(width, width, kernel_size=5, dropout=dropout),
            ResidualConvBlock(width, kernel_size=3, dropout=dropout),
            ResidualConvBlock(width, kernel_size=3, dropout=dropout),
        )
        self.pos_enc = TemporalPositionalEncoding(width, max_len=T + 4)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=width * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    def forward(self, x, return_attn=False):
        x = self.stem(x)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.pos_enc(x)

        if return_attn:
            layer = self.transformer.layers[0]
            x_norm = layer.norm1(x)
            _, attn = layer.self_attn(
                x_norm, x_norm, x_norm,
                need_weights=True,
                average_attn_weights=True,
            )
            self._attn_weights = attn.detach().cpu().numpy()

        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.head(x)

class CNNBaseline(nn.Module):
    def __init__(self, C=11, width=48, dropout=0.15, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(C, width, kernel_size=1),
            ConvBlock1d(width, width, kernel_size=5, dropout=dropout),
            ResidualConvBlock(width, kernel_size=3, dropout=dropout),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.LayerNorm(width),
            nn.Linear(width, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.net(x)

class LSTMBaseline(nn.Module):
    def __init__(self, C=11, hidden=96, dropout=0.15, n_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=C,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 48),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(48, n_classes),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])

class CNNPred2DBaseline(nn.Module):
    def __init__(self, C=11, T=60, n_classes=3, dropout=0.15):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, C)),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=(3, 1)),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=(3, 1)),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.drop = nn.Dropout(dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, C, T)
            flat_dim = self._forward_features(dummy).shape[1]
        self.fc = nn.Linear(flat_dim, n_classes)

    def _forward_features(self, x):
        x = x.transpose(1, 2).unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.pool2(x)
        x = torch.flatten(x, start_dim=1)
        return self.drop(x)

    def forward(self, x):
        return self.fc(self._forward_features(x))

