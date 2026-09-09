from torch import nn
import torch


class Router(nn.Module):
    """
    A simple router for the mixture of experts (MoE) architecture.

    The router takes an input tensor and returns the top-k expert indices and their corresponding probabilities.
    """

    def __init__(self, input_dim, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(input_dim, num_experts)
        self.top_k = top_k

    def forward(self, x):
        gate_logits = self.gate(x)  # (b * num_tokens) x num_experts
        top_k_scores, top_k_indices = torch.topk(
            gate_logits, self.top_k, dim=-1
        )  # (b * num_tokens) x top_k
        top_k_probs = nn.functional.softmax(top_k_scores, dim=-1)
        return top_k_probs, top_k_indices
