import math
import torch
import torch.nn as nn


class OmegaTransformerEdge(nn.Module):
    """OmegaTransformer with geometrically correct kaon-kaon edge bias in attention.

    For each pair (i, j) of kaons, computes true kaon-kaon separation features
    (|Δy_KK|, |Δφ_KK|, ΔR_KK) from absolute kaon positions stored in x_full,
    and maps them to per-head attention biases via a small MLP.

    Edges are symmetric and charge-blind (absolute differences).
    The K-K attention block uses the full MLP output (std=0.01 init, small nonzero
    at epoch 0).  A learnable scalar gate (edge_scale, init=0) multiplies the CLS
    centrality terms only: the CLS token starts blind to edge geometry and opens up
    gradually.  Keeping the K-K block nonzero from epoch 0 is required for correct
    PyTorch mask-merge behavior when src_key_padding_mask is also provided.

    Args:
        y_kaon_idx:   index of kaon rapidity (f_y)  in the full 16-feature tensor
        phi_kaon_idx: index of kaon azimuth  (f_phi) in the full 16-feature tensor
    """

    def __init__(self, in_channels, d_model, nhead, num_layers,
                 dim_feedforward, dropout=0.1, y_kaon_idx=14, phi_kaon_idx=15):
        super().__init__()
        self.nhead   = nhead
        self.y_idx   = y_kaon_idx
        self.phi_idx = phi_kaon_idx

        self.input_proj = nn.Linear(in_channels, d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # 3 edge features → one additive bias per attention head
        self.edge_mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, nhead),
        )
        # Near-zero init on output layer: K-K bias ~0.04 at epoch 0 (small but nonzero)
        nn.init.normal_(self.edge_mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.edge_mlp[-1].bias)
        # Gate for CLS centrality terms: starts at 0 so CLS sees no edge signal at epoch 0
        self.edge_scale = nn.Parameter(torch.zeros(1))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

    def forward(self, x_node, x_full, padding_mask=None):
        # x_node: [B, N, in_channels] — normalised node features for transformer
        # x_full: [B, N, 16]          — full raw feature tensor for position extraction
        # B = batch size, N = number of kaons in event
        B, N, _ = x_node.shape

        # ── Geometrically correct kaon-kaon edge features ─────────────────────
        y_K   = x_full[:, :, self.y_idx]    # [B, N]
        phi_K = x_full[:, :, self.phi_idx]  # [B, N]

        # Outer difference: result[b, i, j] = coord[b, i] − coord[b, j]
        dy_kk   = torch.abs(y_K.unsqueeze(2) - y_K.unsqueeze(1))    # |y_Ki − y_Kj|, [B, N, N]
        # wrap angular gap to (−π, π] then take |·| → [0, π]
        dphi_kk = torch.abs(
            (phi_K.unsqueeze(2) - phi_K.unsqueeze(1) + math.pi) % (2 * math.pi) - math.pi
        )                                                             # |φ_Ki − φ_Kj|, [B, N, N]
        dr_kk   = torch.sqrt(dy_kk**2 + dphi_kk**2 + 1e-8)          # ΔR_KK,          [B, N, N]

        edge_feat     = torch.stack([dy_kk, dphi_kk, dr_kk], dim=-1) # [B, N, N, 3]
        edge_bias_raw = self.edge_mlp(edge_feat)                      # [B, N, N, nhead]

        # CLS centrality: mean edge influence each kaon receives from all others
        # kaon_central[b, h, i] = mean_j edge_bias_raw[b, i, j, h]
        kaon_central = edge_bias_raw.mean(dim=2)                      # [B, N, nhead]
        kaon_central = kaon_central.permute(0, 2, 1)                  # [B, nhead, N]
        kaon_central = kaon_central.reshape(B * self.nhead, N)        # [B*nhead, N]

        # Reshape kaon-kaon block to [B*nhead, N, N]; CLS gets zero-init bias via gate
        edge_bias = edge_bias_raw.permute(0, 3, 1, 2).reshape(B * self.nhead, N, N)

        # Build full [B*nhead, N+1, N+1] bias:
        #   K-K block: small nonzero values from std=0.01 MLP init (never all-zeros)
        #   CLS rows:  kaon_central * edge_scale — starts at 0, opens with gate
        attn_bias = torch.zeros(B * self.nhead, N + 1, N + 1,
                                device=x_node.device, dtype=x_node.dtype)
        attn_bias[:, 1:, 1:] = edge_bias                              # kaon-kaon block
        attn_bias[:, 0, 1:]  = kaon_central * self.edge_scale         # CLS → kaon
        attn_bias[:, 1:, 0]  = kaon_central * self.edge_scale         # kaon → CLS (symmetric)

        # ── Standard forward ──────────────────────────────────────────────────
        h   = self.input_proj(x_node)
        cls = self.cls_token.expand(B, -1, -1)
        h   = torch.cat([cls, h], dim=1)   # [B, 1+N, d_model]

        if padding_mask is not None:
            cls_mask     = torch.zeros(B, 1, dtype=torch.bool, device=padding_mask.device)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)
            padding_mask = padding_mask.float().masked_fill(padding_mask, float('-inf'))

        h = self.transformer(h, mask=attn_bias, src_key_padding_mask=padding_mask)
        return self.classifier(h[:, 0])


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
            padding_mask = padding_mask.float().masked_fill(padding_mask, float('-inf'))

        h = self.transformer(h, src_key_padding_mask=padding_mask)

        # Classify from CLS token
        return self.classifier(h[:, 0])
