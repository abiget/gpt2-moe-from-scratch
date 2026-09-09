import torch
import torch.nn as nn

from gpt2_moe.transformer import TransformBlock
from gpt2_moe.layer_norm import LayerNorm


class GPTModel(nn.Module):

    def __init__(self, cfg):
        """
        seq_len = actual tokens you currently feed (can be smaller)
        context_length = maximum supported length (model design limit)
        """
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.ModuleList(
            [TransformBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.current_pos = 0
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx, use_cache=False):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        # pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        if use_cache:
            pos_ids = torch.arange(
                self.current_pos,
                self.current_pos + seq_len,
                device=in_idx.device,
                dtype=torch.long,
            )
        else:
            pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)

        pos_embeds = self.pos_emb(pos_ids)

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        for block in self.trf_blocks:
            x = block(x, use_cache=use_cache, ptr_current_pos=self.current_pos)
        x = self.final_norm(x)
        logits = self.out_head(x)
        if use_cache:
            self.current_pos += seq_len
        return logits

    def reset_kv_cache(self):
        for block in self.trf_blocks:
            block.att.reset_cache()
        self.current_pos = 0
