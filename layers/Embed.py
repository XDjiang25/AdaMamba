import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
from torch import Tensor


class ReplicationPad1d(nn.Module):
    def __init__(self, padding) -> None:
        super(ReplicationPad1d, self).__init__()
        self.padding = padding

    def forward(self, input: Tensor) -> Tensor:
        replicate_padding = input[:, :, -1].unsqueeze(-1).repeat(1, 1, self.padding[-1])
        output = torch.cat([input, replicate_padding], dim=-1)
        return output
    

class MultiScalePatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_lens, stride, dropout):
        super().__init__()
        self.embedders = nn.ModuleList([
            PatchEmbeddingNG(d_model, p, stride, dropout) for p in patch_lens
        ])
        self.patch_lens = patch_lens
        self.stride = stride

    def forward(self, x):
        """
        x: [B, L, N]
        Returns:
            enc_out: [B*N, total_patch_num, d_model]
            n_vars: N
        """
        B, L, N = x.shape
        all_patch = []
        for embed in self.embedders:
            # patch, n_vars = embed(x.permute(0, 2, 1))  # [B*N, patch_num, d_model]
            patch, n_vars = embed(x)  # [B*N, patch_num, d_model]
            all_patch.append(patch)
        out = torch.cat(all_patch, dim=1)  # concat along patch_num
        return out, n_vars
    

class PatchEmbeddingNG(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbeddingNG, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = ReplicationPad1d((0, stride))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model)

        # Positional embedding
        # self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]

        x = self.padding_patch_layer(x)

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x)
        return self.dropout(x), n_vars