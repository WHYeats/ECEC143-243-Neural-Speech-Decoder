import torch
from torch import nn
import math
from .augmentations import GaussianSmoothing, TemporalMasking, LearnablePatchMasking
from .model_Transformer import PositionalEncoding, RelPosCausalEncoderLayer




class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ConformerConvModule(nn.Module):
    """
    Conformer Conv Module
    """

    def __init__(self, dim, expansion_factor=2, kernel_size=31, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.pointwise_conv1 = nn.Conv1d(dim, dim * expansion_factor * 2, kernel_size=1)
        self.glu = nn.GLU(dim=1)  
        self.depthwise_conv = nn.Conv1d(
            dim * expansion_factor,
            dim * expansion_factor,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim * expansion_factor, 
        )
        self.batch_norm = nn.BatchNorm1d(dim * expansion_factor)
        self.swish = Swish()
        self.pointwise_conv2 = nn.Conv1d(dim * expansion_factor, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  
        x = self.pointwise_conv1(x)
        x = self.glu(x)  
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2) 
        x = self.dropout(x)
        return x + residual


class ConformerFeedForward(nn.Module):

    def __init__(self, dim, expansion_factor=4, dropout=0.1):
        super().__init__()
        inner_dim = dim * expansion_factor
        self.layer_norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, inner_dim)
        self.swish = Swish()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(inner_dim, dim)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.swish(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return residual + 0.5 * x


class ConformerBlock(nn.Module):

    def __init__(
        self,
        dim,
        nhead=8,
        ff_expansion_factor=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        dropout=0.1,
    ):
        super().__init__()
        self.ff1 = ConformerFeedForward(dim, ff_expansion_factor, dropout)

        self.self_attn_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,  # (B, T, D)
        )
        self.self_attn_dropout = nn.Dropout(dropout)

        self.conv_module = ConformerConvModule(
            dim,
            expansion_factor=conv_expansion_factor,
            kernel_size=conv_kernel_size,
            dropout=dropout,
        )

        self.ff2 = ConformerFeedForward(dim, ff_expansion_factor, dropout)
        self.final_layer_norm = nn.LayerNorm(dim)

    def forward(self, x, src_key_padding_mask=None, attn_mask=None):
        """
        x: (B, T, D)
        src_key_padding_mask: (B, T) 
        """
        # FFN 1
        x = self.ff1(x)

        # Self-Attention
        residual = x
        y = self.self_attn_norm(x)
        attn_out, _ = self.self_attn(
            y, y, y, 
            key_padding_mask=src_key_padding_mask, 
            need_weights=False, 
            attn_mask=attn_mask
        )
        attn_out = self.self_attn_dropout(attn_out)
        x = residual + attn_out

        # Conv module
        x = self.conv_module(x)

        # FFN 2
        x = self.ff2(x)

        x = self.final_layer_norm(x)
        return x


class ConformerEncoder(nn.Module):
    """
    ConformerEncoder based on nn.TransformerEncoder
    """

    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        n_head,
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
        self.n_classes = n_classes
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.causal = causal
        self.max_relative_position = 1000
        self.rel_pos_bias = nn.Parameter(
            torch.zeros(2 * self.max_relative_position - 1)
        )

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
            thisLayer = getattr(self, "inpLayer" + str(x))
            thisLayer.weight = torch.nn.Parameter(
                thisLayer.weight + torch.eye(neural_dim)
            )

        self.input_proj = nn.Linear(neural_dim * self.kernelLen, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        self.pos_encoder = PositionalEncoding(d_model=hidden_dim, max_len=1000)

        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    dim=hidden_dim,
                    nhead=n_head,
                    ff_expansion_factor=4,
                    conv_expansion_factor=2,
                    conv_kernel_size=31,
                    dropout=dropout,
                )
                for _ in range(layer_dim)
            ]
        )

        self.fc_decoder_out = nn.Linear(hidden_dim, n_classes + 1)  # +1 for CTC blank
    def _build_attention_mask(self, T, device):

        pos = torch.arange(T, device=device)
        rel = pos[:, None] - pos[None, :]          

        max_rel = self.max_relative_position
        rel_clipped = rel.clamp(-max_rel + 1, max_rel - 1)
        rel_index = rel_clipped + (max_rel - 1)    

        B = self.rel_pos_bias[rel_index]          

        if self.causal:
            causal = torch.triu(
                torch.full((T, T), float("-inf"), device=device),
                diagonal=1,
            )
        else:
            causal = torch.zeros((T, T), device=device)

        return B + causal                          # (T, T)

    def forward(self, neuralInput, dayIdx):

        neuralInput = torch.permute(neuralInput, (0, 2, 1))      # (B, C, T)
        
        # if self.temporalMasking is not None and not self.learnablePatchMasking:
        #     neuralInput = self.temporalMasking(neuralInput)      # (B, C, T)
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
            x = self.PatchMasking(x)
        # x = self.pos_encoder(x)                    # (B, T_out, hidden_dim)
        B, T_out, _ = x.shape
        attn_mask = self._build_attention_mask(T_out, x.device)

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=None, attn_mask=attn_mask)   # (B, T_out, hidden_dim)

        seq_out = self.fc_decoder_out(x)              # (B, T_out, n_classes+1)
        return seq_out