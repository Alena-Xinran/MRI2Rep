import torch
import torch.nn as nn
import math
from .config import Config, VOCAB_SIZE, PAD, BOS, EOS


class ConvBackbone3D(nn.Module):
    """
    Extracts spatial features from 3D volumes.
    Input:  (B, 3, H, W, D)  — channels: [ART, PV, MASK]
    Output: (B, out_ch, H/8, W/8, D/8)
    """
    def __init__(self, in_ch=3, base=32, out_ch=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, base,    3, padding=1),              nn.InstanceNorm3d(base),    nn.GELU(),
            nn.Conv3d(base,   base*2, 3, stride=2, padding=1),   nn.InstanceNorm3d(base*2),  nn.GELU(),
            nn.Conv3d(base*2, base*4, 3, stride=2, padding=1),   nn.InstanceNorm3d(base*4),  nn.GELU(),
            nn.Conv3d(base*4, out_ch, 3, stride=2, padding=1),   nn.InstanceNorm3d(out_ch),  nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for the text decoder."""
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class MRIReportGenerator(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # ── 1. Visual backbone ────────────────────────────────────────────────
        self.backbone = ConvBackbone3D(in_ch=3, base=cfg.backbone_base, out_ch=cfg.d_model)

        # Spatial pool: 24×24×12 → vis_pool_size (default 8×8×4 = 256 tokens)
        # Reduces cross-attention cost from 6912 → 256 tokens.
        self.vis_pool = nn.AdaptiveAvgPool3d(cfg.vis_pool_size)
        vis_tokens = cfg.vis_pool_size[0] * cfg.vis_pool_size[1] * cfg.vis_pool_size[2]

        # Visual Transformer Encoder + learnable positional embedding.
        # Both are disabled when enc_layers=0 (matches old checkpoints without vis_encoder).
        if cfg.enc_layers > 0:
            self.vis_pos_emb = nn.Parameter(torch.randn(1, vis_tokens, cfg.d_model) * 0.02)
            vis_enc_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model, nhead=cfg.nheads,
                dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout,
                batch_first=True, activation="gelu",
                norm_first=True,   # Pre-LN: more stable training
            )
            self.vis_encoder = nn.TransformerEncoder(vis_enc_layer, num_layers=cfg.enc_layers)
        else:
            self.vis_pos_emb = None
            self.vis_encoder = None

        # ── 2. Text decoder (Pre-LN for stable convergence) ──────────────────
        dec_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model, nhead=cfg.nheads,
            dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout,
            batch_first=True, activation="gelu",
            norm_first=True,   # Pre-LN
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg.dec_layers)

        # ── 3. Embeddings & head ──────────────────────────────────────────────
        self.tok_emb = nn.Embedding(VOCAB_SIZE, cfg.d_model, padding_idx=PAD)
        self.pos_emb = PositionalEncoding(cfg.d_model, max_len=cfg.max_seq_len + 50)
        self.head    = nn.Linear(cfg.d_model, VOCAB_SIZE)

        # Auxiliary binary head: "does this scan have any lesion?"
        # Uses mean+max pooling: mean captures global context,
        # max captures the strongest local response (better for detection).
        self.aux_lesion_head = nn.Linear(cfg.d_model * 2, 1)

        # ── 4. Weight initialisation ─────────────────────────────────────────
        self.apply(self._init_weights)
        # Re-init vis_pos_emb if it exists (apply() may have zeroed it)
        if self.vis_pos_emb is not None:
            nn.init.trunc_normal_(self.vis_pos_emb, std=0.02)

    # ── Initialisation ────────────────────────────────────────────────────────
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.Conv3d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # ── Image encoding ────────────────────────────────────────────────────────
    def encode_image(self, img):
        """
        img: (B, 3, H, W, D)
        Returns:
            memory:       (B, vis_tokens, d_model)  after vis_encoder
            backbone_mean:(B, d_model)               CNN features before vis_encoder
        """
        feats  = self.backbone(img)          # (B, d_model, H/8, W/8, D/8)
        feats  = self.vis_pool(feats)        # (B, d_model, ph, pw, pd)
        B, C, ph, pw, pd = feats.shape
        memory = feats.view(B, C, -1).permute(0, 2, 1)   # (B, ph*pw*pd, d_model)
        backbone_feat = torch.cat([
            memory.mean(dim=1),          # (B, d_model) global context
            memory.max(dim=1).values,    # (B, d_model) strongest local response
        ], dim=-1)                       # (B, 2*d_model)
        if self.vis_pos_emb is not None:
            memory = memory + self.vis_pos_emb            # add learnable pos emb
        if self.vis_encoder is not None:
            memory = self.vis_encoder(memory)             # visual self-attention
        return memory, backbone_feat

    # ── Training forward ──────────────────────────────────────────────────────
    def forward(self, img, tgt_seq):
        """
        img:     (B, 3, H, W, D)
        tgt_seq: (B, L)   input tokens (starts with BOS)
        Returns:
            logits:    (B, L, VOCAB_SIZE)
            aux_logit: (B,)  binary has-lesion prediction
        """
        memory, backbone_feat = self.encode_image(img)              # backbone runs once
        aux_logit = self.aux_lesion_head(backbone_feat).squeeze(-1) # (B,) mean+max pooling

        tgt = self.tok_emb(tgt_seq) * math.sqrt(self.cfg.d_model)
        tgt = self.pos_emb(tgt)

        L        = tgt.size(1)
        tgt_mask = torch.triu(torch.ones(L, L, device=img.device, dtype=torch.bool), diagonal=1)

        out    = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
        logits = self.head(out)                                     # (B, L, V)
        return logits, aux_logit

    # ── Inference ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(self, img, max_len=20):
        """Greedy auto-regressive decoding."""
        B      = img.size(0)
        device = img.device

        memory, _ = self.encode_image(img)
        tokens   = torch.full((B, 1), BOS, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            tgt = self.tok_emb(tokens) * math.sqrt(self.cfg.d_model)
            tgt = self.pos_emb(tgt)

            # Causal mask — must match training to avoid train/inference mismatch
            L        = tgt.size(1)
            tgt_mask = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)

            out        = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
            next_logit = self.head(out[:, -1, :])                   # (B, V)
            next_tok   = next_logit.argmax(dim=-1, keepdim=True)    # (B, 1)

            is_eos   = next_tok.squeeze(-1) == EOS
            next_tok = torch.where(finished.unsqueeze(-1),
                                   torch.tensor(PAD, device=device), next_tok)
            finished = finished | is_eos
            tokens   = torch.cat([tokens, next_tok], dim=1)

            if finished.all():
                break

        return tokens
