import torch
import torch.nn as nn


class OmegaTransformer(nn.Module):
    def __init__(self, in_channels, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(in_channels, d_model)

        # Learned CLS token — its output is used for classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

    def forward(self, x, padding_mask=None):
        # x: (batch, seq_len, in_channels)
        batch_size = x.shape[0]

        h = self.input_proj(x)  # (batch, seq_len, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)
        h = torch.cat([cls, h], dim=1)  # (batch, 1+seq_len, d_model)

        # Extend padding mask — CLS token is never masked
        if padding_mask is not None:
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=padding_mask.device)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        h = self.transformer(h, src_key_padding_mask=padding_mask)

        # Classify from CLS token
        return self.classifier(h[:, 0])
