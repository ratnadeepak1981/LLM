
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# NESAMANI LLM - BASELINE EVALUATION
# Loads the existing checkpoint without training.
# ============================================================

CHECKPOINT_PATH = "my_llm.pth"
DEVICE = "cpu"
MAX_NEW_TOKENS = 200

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

config = checkpoint["config"]

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
            "tril", torch.tril(torch.ones(block_size, block_size))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, T, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1)
        wei = wei * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ self.value(x)


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
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        _, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        positions = torch.arange(T, device=idx.device)
        x = tok_emb + self.position_embedding_table(positions)
        x = self.blocks(x)
        return self.lm_head(self.ln_f(x))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx


def encode(text):
    unsupported = sorted(set(text) - set(stoi))
    if unsupported:
        print("  Unsupported characters skipped:", repr("".join(unsupported)))
    return [stoi[ch] for ch in text if ch in stoi]


def decode(token_ids):
    return "".join(itos[int(i)] for i in token_ids)


model = GPT().to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

questions = [
    "What is a purchase order?",
    "What is a stock balance?",
    "What is a tokenizer?",
    "Why do programmers test software?"
]

print("=" * 60)
print("NESAMANI LLM - BASELINE EVALUATION")
print("Checkpoint:", CHECKPOINT_PATH)
print("Device:", DEVICE)
print("=" * 60)

for question in questions:
    input_ids = encode(question)

    print(f"\nQuestion: {question}")

    if not input_ids:
        print("Answer: No supported input characters.")
        continue

    context = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)

    # Reproducible sampling for this baseline run.
    torch.manual_seed(1337)

    with torch.no_grad():
        output = model.generate(context, MAX_NEW_TOKENS)

    answer = decode(output[0, len(input_ids):].tolist())
    print("Generated continuation:", repr(answer))

print("\nEvaluation complete. No model training was performed.")
