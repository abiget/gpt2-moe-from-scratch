import argparse
import torch
import tiktoken
from pathlib import Path

from gpt2_moe.gpt_model import GPTModel
from utils.data_loader import create_dataloader
from utils.data_splitter import split_data_samples
from utils.train import train_model_simple
from utils.utils import plot_losses

SEED = 123
DATA_FILE = Path("data/en.wiki.train.tokens")
CHECKPOINTS_DIR = Path("checkpoints")
FIGURES_DIR = Path("figures")

torch.manual_seed(SEED)


def main(args):
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False,
        "num_experts": 4,
        "top_k": 2,
    }

    tokenizer = tiktoken.get_encoding("gpt2")

    train_data, val_data = split_data_samples(tokenizer=tokenizer, file_path=DATA_FILE)

    train_loader = create_dataloader(
        train_data,
        batch_size=args.batch_size,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = create_dataloader(
        val_data,
        batch_size=args.batch_size,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.1
    )
    num_epochs = args.num_epochs
    train_losses, val_losses, token_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context="Every effort moves you",
        tokenizer=tokenizer,
    )

    CHECKPOINTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(
        epochs_tensor,
        token_seen,
        train_losses,
        val_losses,
        save_path=FIGURES_DIR / "loss.png",
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        CHECKPOINTS_DIR / "model_and_optimizer_pretrain.pth",
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Pretrain a GPT model.")
    parser.add_argument(
        "--batch_size", type=int, default=6, help="Batch size for training."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=4, help="Number of epochs for training."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0004,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of workers for data loading."
    )
    args = parser.parse_args()
    main(args)
