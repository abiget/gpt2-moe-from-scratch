def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    return text_data


def split_data_samples(
    tokenizer,
    file_path="en.wiki.train.tokens",
    train_size=0.9,
):
    text_data = load_data(file_path)

    train_ratio = train_size
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))
    print("Characters:", total_characters)
    print("Tokens:", total_tokens)

    return train_data, val_data
