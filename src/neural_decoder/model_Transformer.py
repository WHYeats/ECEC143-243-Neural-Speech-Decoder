import torch
from torch import nn
import math
from .augmentations import GaussianSmoothing, TemporalMasking, LearnablePatchMasking

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (B, T, d_model)
        """
        T = x.size(1)

        return x + self.pe[:, :T]
    
class RelPosCausalEncoderLayer(nn.Module):
    """
    Pre-Norm Transformer Encoder Layer with:
      - Causal Mask M 
      - Relative Position Bias B 
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.0,
        max_relative_position=1000,
        causal = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.max_relative_position = max_relative_position
        self.causal = causal

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        # === Pre-Norm ===
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

        # === Learnable Relative Position Bias b ∈ R^{2L-1} ===
        self.rel_pos_bias = nn.Parameter(torch.zeros(2 * max_relative_position - 1))

    # ----------- 构造 B + M ----------
    def _build_attention_mask(self, T, device):
        """
        Return (T, T) attention mask: B + causal M
        """
        # Relative positions: i - j
        pos = torch.arange(T, device=device)
        rel = pos[:, None] - pos[None, :]  # (T, T)

        # Clip to [-L+1, L-1]
        max_rel = self.max_relative_position
        rel_clipped = rel.clamp(-max_rel + 1, max_rel - 1)

        # Shift index to [0, 2L-2]
        rel_index = rel_clipped + (max_rel - 1)

        # B_{i,j}
        B = self.rel_pos_bias[rel_index]  # (T, T)

        if self.causal:
            causal = torch.triu(
                torch.full((T, T), float("-inf"), device=device),
                diagonal=1,
            )
        else:
            causal = torch.zeros((T, T), device=device)

        return B + causal  # (T, T)

    # ----------- Forward ----------
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        src: (B, T, D)
        """
        B, T, D = src.shape
        device = src.device

        attn_mask = self._build_attention_mask(T, device)

        # === Pre-Norm Self-Attention ===
        src2 = self.norm1(src)
        attn_out, _ = self.self_attn(
            src2,
            src2,
            src2,
            attn_mask=attn_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        src = src + self.dropout1(attn_out)

        # === Pre-Norm Feed Forward ===
        src2 = self.norm2(src)
        ff = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(ff)

        return src

    
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        nhead,
        layer_dim,
        nDays=24,
        dropout=0,
        device="cuda",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        temporalMaxLen=40,
        temporalMaskP=0.5,
        temporalMaskValue=0.0,
        temporalNumMask=2,
        patchMaskRatio=0.05,
        patchNumMask=2,
        causal = True,

    ):
        super().__init__()

        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.nhead = nhead
        self.n_classes = n_classes
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.inputLayerNonlinearity = torch.nn.Softsign()

        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )

        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))

        if temporalMaxLen > 0:
            self.temporalMasking = TemporalMasking(
                max_mask_length=temporalMaxLen,
                n_masks=temporalNumMask,
                mask_value=temporalMaskValue,
            )
        else:
            self.temporalMasking = None

        if patchMaskRatio > 0:
            self.PatchMasking = LearnablePatchMasking(
                n_masks=patchNumMask,
                mask_ratio=patchMaskRatio,
                hidden_dim=hidden_dim,
            )
        else:
            self.PatchMasking = None

        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)

        for x in range(nDays):
            setattr(self, "inpLayer" + str(x), nn.Linear(neural_dim, neural_dim))

        for x in range(nDays):
            thisLayer = getattr(self, "inpLayer" + str(x))
            thisLayer.weight = torch.nn.Parameter(
                thisLayer.weight + torch.eye(neural_dim)
            )

        self.input_proj = nn.Linear(neural_dim * self.kernelLen, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        self.pos_encoder = PositionalEncoding(d_model=hidden_dim, max_len=1000)

        # TransformerEncoder
        encoder_layer = RelPosCausalEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            max_relative_position=1000,
            causal = causal,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layer_dim,
        )
        self.post_norm = nn.LayerNorm(hidden_dim)
        self.fc_decoder_out = nn.Linear(hidden_dim, n_classes + 1)  # +1 for CTC blank

    def forward(self, neuralInput, dayIdx):

        neuralInput = torch.permute(neuralInput, (0, 2, 1))      # (B, C, T)    
        if self.temporalMasking is not None:
            neuralInput = self.temporalMasking(neuralInput)      # (B, C, T)

        neuralInput = self.gaussianSmoother(neuralInput)         # (B, C, T)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))      # (B, T, C)

        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)   # (B, D, D)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)               # (B, T, D)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        stridedInputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )  

        x = self.input_proj(stridedInputs)         # (B, T_out, hidden_dim)
        x = self.input_norm(x)                     # (B, T_out, hidden_dim)
        
        if self.PatchMasking is not None:
            x = self.PatchMasking(x)               # (B, T_out, hidden_dim)
            
        # x = self.pos_encoder(x)                    # (B, T_out, hidden_dim)

        hid = self.transformer(x)                  # (B, T_out, hidden_dim)
        
        hid = self.post_norm(hid)                  # (B, T_out, hidden_dim)
        seq_out = self.fc_decoder_out(hid)         # (B, T_out, n_classes+1)
        return seq_out    