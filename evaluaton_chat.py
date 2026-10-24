
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

CHECKPOINT_PATH = Path("my_llm_experiment.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if not CHECKPOINT_PATH.exists():
    raise SystemExit(f"Checkpoint not found: {CHECKPOINT_PATH}")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

stoi = checkpoint["stoi"]
itos = checkpoint["itos"]
config = checkpoint["config"]

vocab_size = config["vocab_size"]
block_size = config["block_size"]
n_embd = config["n_embd"]
n_head = config["n_head"]
n_layer = config["n_layer"]
dropout = config["dropout"]


def encode(text):
    unknown = [ch for ch in text if ch not in stoi]
    if unknown:
        raise ValueError(
            f"Question contains characters outside the vocabulary: {unknown}"
        )
    return [stoi[ch] for ch in text]


def decode(ids):
    return "".join(itos[int(i)] for i in ids)


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, T, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1)
        wei = wei * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
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
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),
            nn.GELU(),
            nn.Linear(4 * embedding_size, embedding_size),
            nn.Dropout(dropout),
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
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        _, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        positions = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(positions)
        x = self.blocks(tok_emb + pos_emb)
        x = self.ln_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=180, temperature=0.8):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)[:, -1, :]

            # Temperature controls the randomness of sampling.
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


model = GPT().to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

questions = [
    "What is a tokenizer?",
    "What is an embedding?",
    "What is attention in a Transformer?",
    "What is a training dataset?",
    "What is a validation dataset?",
    "What is a loss function?",
    "What is an optimizer?",
    "What is next-character prediction?",
    "What is a goods receipt?",
    "What is a sales order?",
    "What is a stock transfer?",
]

print("=" * 60)
print("NESAMANI LLM - NEW Q&A EVALUATION")
print("Checkpoint:", CHECKPOINT_PATH)
print("=" * 60)

for question in questions:
    # The model predicts the continuation after the question.
    prompt = question + "\n"

    try:
        input_ids = torch.tensor(
            [encode(prompt)], dtype=torch.long, device=DEVICE
        )
    except ValueError as error:
        print(f"\nQuestion: {question}")
        print("Cannot test:", error)
        continue

    generated = model.generate(
        input_ids,
        max_new_tokens=180,
        temperature=0.8
    )

    full_text = decode(generated[0].tolist())
    continuation = full_text[len(prompt):]

    # Stop display at the next blank-line-separated example.
    answer = continuation.split("\n\n", 1)[0].strip()

    print(f"\nQuestion: {question}")
    print("Generated answer:", answer)

print("\nEvaluation finished.")
