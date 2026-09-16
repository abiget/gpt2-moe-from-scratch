import torch

from utils.loss_function import calc_loss_loader
from utils.utils import text_to_token_ids, token_ids_to_text


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    # dropout is disabled during evaluation for stable, reproducible results
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)

    model.train()
    return train_loss, val_loss


def generate(
    model,
    idx,
    max_new_tokens,
    context_size,
    temperature=0.0,
    top_k=None,
    eos_id=None,
    use_cache=True,
):
    model.eval()
    idx = idx[:, -context_size:]
    if use_cache:
        # Init cache with full prompt
        model.reset_kv_cache()

    # generated = []

    with torch.no_grad():
        # prefill the cache
        if use_cache:
            logits = model(idx, use_cache=True)
        else:
            logits = model(idx, use_cache=False)

        logits = logits[:, -1, :]
        idx_next = sample_next_token(logits, temperature, top_k)

        for _ in range(max_new_tokens):

            if eos_id is not None and idx_next == eos_id:
                break

            # generated.append(idx_next)
            idx = torch.cat((idx, idx_next), dim=1)

            if use_cache:
                logits = model(idx_next, use_cache=True)
            else:
                idx_cond = idx[:, -context_size:]
                logits = model(idx_cond, use_cache=False)

            logits = logits[:, -1, :]
            idx_next = sample_next_token(logits, temperature, top_k)

    return idx


def sample_next_token(logits, temperature=1.0, top_k=None):
    if top_k is not None:
        top_logits, _ = torch.topk(logits, top_k, dim=-1)
        min_val = top_logits[:, -1]
        logits = torch.where(
            condition=logits < min_val,
            input=torch.tensor(float("-inf")).to(logits.device),
            other=logits,
        )

    if temperature > 0.0:
        logits = logits / temperature
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probas, num_samples=1)
    else:
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)

    return idx_next


def generate_and_print_sample(model, tokenizer, device, start_context, use_cache=False):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():
        token_ids = generate(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
            use_cache=use_cache,
        )

    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", ""))
    model.train()
