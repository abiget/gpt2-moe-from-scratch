import torch
from torch import nn


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )

    return loss


def moe_load_balance_loss_batch(router_logits, top_k_indices, num_experts):
    """
    Calculate the load balancing loss for a batch of data.

    Args:
        router_logits (torch.Tensor): The logits from the router, shape (batch_size * num_tokens, num_experts).
        top_k_indices (torch.Tensor): The top-k expert indices from the router, shape (batch_size * num_tokens, top_k).
        num_experts (int): The total number of experts.

    Returns:
        torch.Tensor: The load balancing loss for the batch.

    Equation: load_balance_loss = num_experts * sum(mean(expert_probs) * expert_load)
    """

    # calculate the load for each expert
    # (num_experts,)
    T, N = top_k_indices.shape
    # (T, N, num_experts)
    mask = nn.functional.one_hot(top_k_indices, num_classes=num_experts).float()

    # calculate the load balancing loss
    # (batch_size * num_tokens, num_experts)
    expert_probs = torch.softmax(router_logits, dim=-1)
    # (num_experts,)
    expert_load = mask.sum(dim=(0, 1)) / (T * N)
    expert_prob_avg = expert_probs.mean(dim=0)
    # (num_experts,)
    return num_experts * torch.sum(expert_prob_avg * expert_load)


def router_z_loss_batch(router_logits):
    """
    Calculate the router z-loss for a batch of data.

    Args:
        router_logits (torch.Tensor): The logits from the router, shape (batch_size * num_tokens, num_experts).
    Returns:
        torch.Tensor: The router z-loss for the batch.

    Equation: z_loss = log(mean(exp(router_logits)))^2
    """

    z_loss = torch.exp(router_logits)  # (batch_size * num_tokens, num_experts)
    z_loss = torch.sum(z_loss, dim=-1)  # (batch_size * num_tokens,)
    z_loss = torch.mean(z_loss)  # scalar
    z_loss = torch.log(z_loss) ** 2  # scalar

    return z_loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss_z = 0.0
            loss_bal = 0.0
            for block in model.trf_blocks:
                loss_bal = loss_bal + moe_load_balance_loss_batch(
                    block.moe.last_gate_logits,
                    block.moe.last_top_k_indices,
                    block.moe.num_experts,
                )

                loss_z = loss_z + router_z_loss_batch(block.moe.last_gate_logits)
            loss = loss + loss_bal + loss_z
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches
