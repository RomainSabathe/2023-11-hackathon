from pathlib import Path
from typing import List, Union, Optional, Tuple

import torch
import requests
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

DATA_DIR = Path("/mnt/data/tiny_gpt/data")
factor = 2
SEQ_LEN = 256
BATCH_SIZE = 64 * factor
LEARNING_RATE = 3e-4 * factor
N_EPOCHS = 5
N_ITER_PER_EPOCH = 500
N_HEADS = 6
N_BLOCKS = 6
HIDDEN_DIM = 384
DROPOUT_RATE = 0.2
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_shakespeare_dataset() -> str:
    save_path = DATA_DIR / "shakespeare.txt"

    if not save_path.exists():
        save_path.parent.mkdir(exist_ok=True, parents=True)
        req = requests.get(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        )
        text = req.text
        save_path.write_text(text)

    return save_path.read_text()


class Tokenizer:
    def fit(self, text) -> None:
        unique_chars = sorted(list(set(text)))
        self._char_to_idx = {char: idx for (idx, char) in enumerate(unique_chars)}
        self._idx_to_char = {idx: char for (idx, char) in enumerate(unique_chars)}

    def to_idx(self, text: Union[str, List[str]]) -> torch.Tensor:
        if not hasattr(self, "_char_to_idx"):
            raise AttributeError("You must use .fit() before calling `to_idx()`.")
        if isinstance(text, str):
            text = [text]

        sentences = text
        to_return = torch.tensor(
            [[self._char_to_idx[char] for char in sentence] for sentence in sentences],
            dtype=torch.long,
        )
        return to_return

    def to_text(self, idx: torch.Tensor) -> List[str]:
        # idx is (B, T)
        if not hasattr(self, "_idx_to_char"):
            raise AttributeError("You must use .fit() before calling `to_char()`.")
        if not isinstance(idx, torch.Tensor):
            idx = torch.tensor(idx, dtype=torch.long)

        idx = idx.tolist()
        sentences = []
        for sentence in idx:
            sentences.append("".join([self._idx_to_char[_idx] for _idx in sentence]))
        return sentences

    @property
    def vocab_size(self):
        return len(self._char_to_idx)


def train_val_split(text: str, frac: float = 0.9):
    n = int(len(text) * frac)
    return text[:n], text[n:]


def get_batch(
    dataset: torch.Tensor, batch_size: int = BATCH_SIZE, seq_len: int = SEQ_LEN
):
    # dataset is (N,) and is of type torch.long.
    start_idx = torch.randint(0, len(dataset) - seq_len - 1, [batch_size])
    offsets = torch.arange(seq_len).view(seq_len)
    input_idx = start_idx[:, None] + offsets  # we use broadcasting.
    input_batch = dataset[input_idx]

    target_idx = input_idx + 1
    target_batch = dataset[target_idx]

    return input_batch, target_batch


class BigramModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = None):
        super().__init__()

        if embedding_dim is None:
            embedding_dim = vocab_size
        self.embedder = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, input: torch.Tensor, target: Optional[torch.Tensor] = None):
        # input:  (B, T)
        # target: (B, T)
        logits = self.embedder(input)  # (B, T, C)

        if target is not None:
            # Reshaping to standard classification. We have B*T samples and a probability distribution
            # over C classes.
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            target = target.view(B * T)
            loss = F.cross_entropy(logits, target)
        else:
            loss = None

        return logits, loss

    def generate(self, prompt: torch.Tensor, max_len: int = 30) -> torch.Tensor:
        # prompt is (B, T)
        for _ in range(max_len):
            logits, _ = self(prompt)  # (B, T, C)  with presumably B = 1
            logits_next_token = logits[:, -1, :]
            softmax_next_token = F.softmax(logits_next_token, dim=1)  # softmax over C
            next_token = torch.multinomial(
                softmax_next_token, num_samples=1, replacement=True
            )
            prompt = torch.cat((prompt, next_token), dim=1)  # concat over T

        return prompt


class Head(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, sequence_length: int):
        super().__init__()

        self.key_fn = nn.Linear(in_dim, hidden_dim, bias=False)
        self.query_fn = nn.Linear(in_dim, hidden_dim, bias=False)
        self.value_fn = nn.Linear(in_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(DROPOUT_RATE)

        future_mask = torch.tril(
            torch.ones(sequence_length, sequence_length, device=device)
        )
        self.register_buffer("future_mask", future_mask)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input

        B, T, in_dim = x.shape
        key = self.key_fn(x)  # (B, T, hidden_dim)
        query = self.query_fn(x)  # (B, T, hidden_dim)
        value = self.value_fn(x)  # (B, T, hidden_dim)

        nB, nT, nhidden_dim = 0, 1, 2
        _, _, hidden_dim = key.shape
        attention_matrix = (query @ key.transpose(nhidden_dim, nT)) / torch.sqrt(
            torch.tensor(hidden_dim)
        )  # (B, T, T)
        # a tester: key @ query.T

        future_mask = self.future_mask  # (seq_len, seq_len)  with T <= seq_len
        future_mask = future_mask[:T, :T]  # (T, T)
        attention_matrix = attention_matrix.masked_fill(
            future_mask == 0.0, float("-inf")
        )  # (B, T, T)
        attention_matrix = F.softmax(
            attention_matrix, dim=2
        )  # normalizing so the rows sum to 1.
        attention_matrix = self.dropout(attention_matrix)

        x = (
            attention_matrix @ value
        )  # (B, T, T) @ (B, T, hidden_dim) --> (B, T, hidden_dim)

        return x


class MultiHead(nn.Module):
    def __init__(
        self, hidden_dim: int, head_size: int, sequence_length: int, n_heads: int
    ):
        super().__init__()

        self.attention_heads = nn.ModuleList(
            [
                Head(
                    in_dim=hidden_dim,
                    hidden_dim=head_size,
                    sequence_length=sequence_length,
                ).to(device)
                for _ in range(n_heads)
            ]
        )
        self.linear = nn.Linear(head_size * n_heads, hidden_dim)
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        x = torch.concat(
            [attention_head(x) for attention_head in self.attention_heads], dim=2
        )  # Each attention_head outputs (B, T, hidden_dim//n_heads) --> (B, T, hidden_dim)
        x = self.linear(x)
        x = self.dropout(x)

        return x


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, inflation_factor: int = 4):
        super().__init__()
        self.expansion_fn = nn.Linear(hidden_dim, hidden_dim * inflation_factor)
        self.compression_fn = nn.Linear(hidden_dim * inflation_factor, hidden_dim)
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        x = self.expansion_fn(x)
        x = F.relu(x)
        x = self.compression_fn(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, hidden_dim: int, sequence_length: int, n_heads: int):
        super().__init__()

        self.multi_head_1 = MultiHead(
            hidden_dim=hidden_dim,
            sequence_length=sequence_length,
            head_size=hidden_dim // n_heads,
            n_heads=n_heads,
        )
        self.layer_norm_1 = nn.LayerNorm(hidden_dim)
        self.layer_norm_2 = nn.LayerNorm(hidden_dim)
        self.feed_forward = FeedForward(hidden_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        x = x + self.multi_head_1(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        sequence_length: int,
        n_heads: int,
        n_blocks: int,
    ):
        super().__init__()

        self.sequence_length = sequence_length

        self.vocab_embedder = nn.Embedding(vocab_size, hidden_dim)
        self.positional_embedder = nn.Embedding(sequence_length, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.language_model_head = nn.Linear(hidden_dim, vocab_size)

        self.block = nn.Sequential(
            *[
                Block(
                    hidden_dim=hidden_dim,
                    sequence_length=sequence_length,
                    n_heads=n_heads,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(
        self, input: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input.shape

        x = input  # (B, T)
        vocab_embedding = self.vocab_embedder(x)  # (B, T, hidden_dim)
        positional_embedding = self.positional_embedder(
            torch.arange(T, device=device)
        )  # (T, hidden_dim)
        x = (
            vocab_embedding + positional_embedding
        )  # (B, T, hidden_dim). It works thanks to broadcasting.

        x = self.block(x)

        x = self.layer_norm(x)
        x = self.language_model_head(x)  # (B, T, vocab)
        logits = x  # (B, T, vocab)

        if target is not None:
            # target is (B, T)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            target = target.view(B * T)
            loss = F.cross_entropy(logits, target)
        else:
            loss = None

        return logits, loss

    def generate(self, prompt: torch.Tensor, max_len: int) -> torch.Tensor:
        B, T = prompt.shape
        # Pre-allocating the tensor that will hold the indices of all the output tokens.
        output = torch.cat((prompt, torch.zeros(B, max_len).to(device)), dim=1).long()

        left_idx, right_idx = 0, min(T, self.sequence_length - 1)
        for _ in range(max_len):
            prompt = output[:, left_idx:right_idx]

            logits, _ = self(prompt)  # (B, T, vocab_size) with presumably B=1
            logits_next_token = logits[:, -1, :]  # (B, vocab_size)
            logits_next_token = F.softmax(
                logits_next_token, dim=1
            )  # softmax over vocab_size
            next_token_idx = torch.multinomial(
                logits_next_token, num_samples=1
            )  # (B, 1)
            output[:, right_idx] = next_token_idx

            right_idx += 1
            if (right_idx - left_idx) > self.sequence_length:
                left_idx += 1

        return output


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataset: torch.Tensor,
    eval_iters: int = 200,
    batch_size: int = BATCH_SIZE,
    seq_len: int = SEQ_LEN,
) -> torch.float32:
    losses = torch.zeros(eval_iters)

    model.eval()
    for k in range(eval_iters):
        input, target = get_batch(dataset, batch_size=batch_size, seq_len=seq_len)
        _, loss = model(input, target)
        losses[k] = loss.item()
    model.train()

    return losses.mean()


if __name__ == "__main__":
    text = get_shakespeare_dataset()

    tokenizer = Tokenizer()
    tokenizer.fit(text)
    dataset = tokenizer.to_idx(text).squeeze().to(device)
    train_dataset, val_dataset = train_val_split(dataset)

    input, target = get_batch(train_dataset)
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=HIDDEN_DIM,
        sequence_length=SEQ_LEN,
        n_heads=N_HEADS,
        n_blocks=N_BLOCKS,
    ).to(device)
    # model = BigramModel(vocab_size=tokenizer.vocab_size).to(device)
    logits, loss = model(input, target)
    loss_before_training = evaluate_model(model, val_dataset)
    print(f"{loss_before_training=}")

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    for _ in range(N_EPOCHS):
        for _ in range(N_ITER_PER_EPOCH):
            input, target = get_batch(train_dataset)
            logits, loss = model(input, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        train_loss_during_training = evaluate_model(model, train_dataset)
        val_loss_during_training = evaluate_model(model, val_dataset)
        print(f"{train_loss_during_training=} {val_loss_during_training=}")

    prompt = "Lord, "
    prompt = tokenizer.to_idx(prompt).to(device)
    generated_tokens = model.generate(prompt, max_len=200)
    generated_text = tokenizer.to_text(generated_tokens)[0]
    print(generated_text)

    import ipdb

    ipdb.set_trace()
    pass
