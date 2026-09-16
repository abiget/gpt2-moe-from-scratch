from torch import nn


class Expert(nn.Module):
    """
    A single expert in the mixture of experts (MoE) architecture.

    (Silu(x@W_gate) * (x@W_up))@W_down
    """

    def __init__(self, cfg):
        super().__init__()
        self.W_up = nn.Linear(cfg["emb_dim"], cfg["emb_dim"])
        self.W_gate = nn.Linear(cfg["emb_dim"], cfg["emb_dim"])
        self.W_down = nn.Linear(cfg["emb_dim"], cfg["emb_dim"])
        self.silu = nn.SiLU()

    def forward(self, x):
        x_up = self.W_up(x)  # b x num_tokens x (2 * emb_dim)
        x_gate = self.W_gate(x)  # b x num_tokens x (2 * emb_dim)
        x = self.silu(x_gate) * x_up  # b x num_tokens x (2 * emb_dim)
        x = self.W_down(x)  # b x num_tokens x emb_dim
        return x
