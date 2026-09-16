from torch import nn
import torch
from gpt2_moe.expert import Expert
from gpt2_moe.router import Router


class MoEs(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_experts = cfg["num_experts"]
        self.experts = nn.ModuleList([Expert(cfg) for _ in range(self.num_experts)])
        self.router = Router(cfg["emb_dim"], self.num_experts)

    def forward(self, x):
        batch_size, num_tokens, emb_dim = x.shape
        # (b * num_tokens) x emb_dim
        x_flatten = x.reshape(batch_size * num_tokens, emb_dim)

        # (b * num_tokens) x top_k
        top_k_probs, top_k_indices, gate_logits = self.router(x_flatten)

        if self.training:
            self.last_gate_logits = gate_logits
            self.last_top_k_indices = top_k_indices
        else:
            self.last_gate_logits = None
            self.last_top_k_indices = None

        k = top_k_indices.shape[-1]

        out = torch.zeros_like(x_flatten)  # (b * num_tokens) x emb_dim

        for i in range(k):
            expert_indices = top_k_indices[:, i]  # (b * num_tokens)
            expert_probs = top_k_probs[:, i].unsqueeze(-1)  # (b * num_tokens) x 1

            for expert_id in expert_indices.unique():
                mask = expert_indices == expert_id

                if mask.sum() == 0:
                    continue

                # (num_tokens_expert, emb_dim)
                x_expert = x_flatten[mask]

                # (num_tokens_expert, emb_dim)
                expert_output = self.experts[expert_id](x_expert)

                # (num_tokens_expert, emb_dim)
                out[mask] = out[mask] + expert_probs[mask] * expert_output

        out = out.reshape(batch_size, num_tokens, emb_dim)
        return out
