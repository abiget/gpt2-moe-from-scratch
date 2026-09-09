import torch.nn as nn

from gpt2_moe.feed_forward import FeedForward
from gpt2_moe.multi_head_attention import MultiHeadAttention
from gpt2_moe.layer_norm import LayerNorm
from gpt2_moe.moes import MoEs


class TransformBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.moe = MoEs(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, use_cache=False, ptr_current_pos=0):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, use_cache=use_cache, ptr_current_pos=ptr_current_pos)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.moe(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
