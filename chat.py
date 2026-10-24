
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# NESAMANI LLM - TERMINAL CHAT
# Loads an existing checkpoint. Does not train the model.
# ============================================================

CHECKPOINT_PATH = "my_llm.pth"
DEVICE = "cpu"

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

config = checkpoint["config"]

# Use the exact configuration saved during training.
vocab_size = config["vocab_size"]
block_size = config["block_size"]
n_embd = config["n_embd"]
n_head = config["n_head"]
n_layer = config["n_layer"]
dropout = config["dropout"]

stoi = checkpoint["stoi"]
itos = checkpoint["itos"]


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1)
        wei = wei * (k.shape[-1] ** -0.5)

        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([
            Head(head_size) for _ in range(num_heads)
        ])

        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        return self.dropout(out)


class FeedForward(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),
            nn.GELU(),
            nn.Linear(4 * embedding_size, embedding_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.network(x)


class Block(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super().__init__()

        head_size = embedding_size // num_heads

        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(embedding_size)

        self.ln1 = nn.LayerNorm(embedding_size)
        self.ln2 = nn.LayerNorm(embedding_size)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(
            vocab_size, n_embd
        )

        self.position_embedding_table = nn.Embedding(
            block_size, n_embd
        )

        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head)
            for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02
            )

    def forward(self, idx):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        positions = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(positions)

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]

            logits = self(idx_cond)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# Load the trained weights into the model.
model = GPT().to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def encode(text):
    unknown = sorted(set(text) - set(stoi.keys()))

    if unknown:
        print(
            "Warning: unsupported characters will be skipped:",
            repr("".join(unknown))
        )

    return [stoi[ch] for ch in text if ch in stoi]


def decode(token_ids):
    return "".join(itos[int(token_id)] for token_id in token_ids)


def main():
    print("=" * 50)
    print("NESAMANI LLM - TERMINAL CHAT")
    print("Model:", CHECKPOINT_PATH)
    print("Device:", DEVICE)
    print("Type 'exit' to quit.")
    print("=" * 50)

    while True:
        try:
            question = input("\nYou: ")

            if question.strip().lower() in {"exit", "quit"}:
                print("NESAMANI LLM: Goodbye!")
                break

            if not question.strip():
                continue

            input_ids = encode(question)

            if not input_ids:
                print("NESAMANI LLM: No supported input characters.")
                continue

            context = torch.tensor(
                [input_ids],
                dtype=torch.long,
                device=DEVICE
            )

            with torch.no_grad():
                output = model.generate(
                    context,
                    max_new_tokens=200
                )

            generated_ids = output[0].tolist()

            # Display only the newly generated continuation.
            answer = decode(generated_ids[len(input_ids):])
            print("NESAMANI LLM:", answer)

        except (KeyboardInterrupt, EOFError):
            print("\nNESAMANI LLM: Goodbye!")
            break


if __name__ == "__main__":
    main()
