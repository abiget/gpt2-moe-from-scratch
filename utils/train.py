from utils.loss_function import (
    calc_loss_batch,
    moe_load_balance_loss_batch,
    router_z_loss_batch,
)
from utils.evaluate import evaluate_model
from utils.evaluate import generate_and_print_sample


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, tartget_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, tartget_batch, model, device)

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

            loss.backward()
            optimizer.step()
            # cummulative num of tokens processed so far during training
            tokens_seen += input_batch.numel()
            global_step += 1

            # when global step is multiple of eval_freq
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss: .3f}"
                )

        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen
