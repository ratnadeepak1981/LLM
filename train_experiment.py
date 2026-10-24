
# ============================================================
# NESAMANI LLM - CONTROLLED TRAINING EXPERIMENT
# File: train_experiment.py
# Framework: Python + PyTorch
# Architecture: Character-level GPT / Transformer
#
# Safety:
#   - Reads train.txt
#   - Does not modify train.py
#   - Does not overwrite my_llm.pth
#   - Saves to my_llm_experiment.pth
# ============================================================

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# STEP 1: CONFIGURATION
# Keep the original model architecture and settings.
# ============================================================

torch.manual_seed(1337)

batch_size = 32
block_size = 128
max_iters = 3000
eval_interval = 300
learning_rate = 3e-4
eval_iters = 50

n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.1

device = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH = Path("train.txt")
OUTPUT_PATH = Path("my_llm_experiment.pth")

print("=" * 60)
print("NESAMANI LLM - CONTROLLED EXPERIMENT")
print("=" * 60)
print("Device:", device)
print("Training data:", DATA_PATH)
print("Output checkpoint:", OUTPUT_PATH)


# ============================================================
# STEP 2: LOAD DATA
# ============================================================

if not DATA_PATH.exists():
    raise SystemExit(
        "ERROR: train.txt was not found in the project directory."
    )

text = DATA_PATH.read_text(encoding="utf-8")

print("Dataset characters:", len(text))

minimum_required = (2 * block_size) + 2

if len(text) < minimum_required:
    raise SystemExit(
        f"ERROR: Dataset must contain at least "
        f"{minimum_required} characters."
    )


# ============================================================
# STEP 3: CREATE CHARACTER VOCABULARY
# Each character receives an integer ID.
# ============================================================

chars = sorted(set(text))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

print("Vocabulary size:", vocab_size)


def encode(s):
    """Convert characters to integer IDs."""
    return [stoi[ch] for ch in s]


def decode(ids):
    """Convert integer IDs back to characters."""
    return "".join(itos[int(i)] for i in ids)


# ============================================================
# STEP 4: TRAIN / VALIDATION SPLIT
# Find a paragraph boundary near the 90% target.
# Do not cut directly through a question-answer paragraph.
# ============================================================

target_index = int(0.90 * len(text))

# Find paragraph boundaries before and after the target.
boundaries = [
    i for i in range(1, len(text) - 1)
    if text[i:i + 2] == "\n\n"
]

if not boundaries:
    raise SystemExit(
        "ERROR: No paragraph boundaries found in train.txt."
    )

# Choose the boundary closest to the 90% target.
split_index = min(
    boundaries,
    key=lambda i: abs(i - target_index)
)

train_text = text[:split_index]
val_text = text[split_index + 2:]

if len(train_text) <= block_size:
    raise SystemExit("ERROR: Training data is too small.")

if len(val_text) <= block_size:
    raise SystemExit("ERROR: Validation data is too small.")

# Encode the two portions independently.
train_data = torch.tensor(encode(train_text), dtype=torch.long)
val_data = torch.tensor(encode(val_text), dtype=torch.long)

print("\nDATA SPLIT")
print("-" * 60)
print("Target split position:", target_index)
print("Actual split position:", split_index)
print("Training characters:", len(train_data))
print("Validation characters:", len(val_data))
print("Training proportion:", f"{len(train_data) / len(text):.1%}")
print("Validation proportion:", f"{len(val_data) / len(text):.1%}")
print("Training ends with:", repr(train_text[-80:]))
print("Validation starts with:", repr(val_text[:80]))

# This check helps identify which QA examples occur in training.
questions_to_check = [
    "What is a purchase order?",
    "What is a stock balance?",
    "Why do programmers test software?",
    "What is a backup?",
    "How can a business manage its products?",
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

print("\nQUESTION LOCATION CHECK")
print("-" * 60)

for question in questions_to_check:
    position = text.find(question)

    if position == -1:
        location = "NOT FOUND"
    elif position < split_index:
        location = "TRAINING"
    else:
        location = "VALIDATION"

    print(f"{location:12} | {question}")


# ============================================================
# STEP 5: BATCH FUNCTION
# x contains input characters.
# y contains the same sequence shifted one character ahead.
# ============================================================

def get_batch(split):
    if split == "train":
        data_source = train_data
    else:
        data_source = val_data

    max_start = len(data_source) - block_size - 1

    starts = torch.randint(
        0,
        max_start + 1,
        (batch_size,)
    )

    x = torch.stack([
        data_source[i:i + block_size]
        for i in starts.tolist()
    ])

    y = torch.stack([
        data_source[i + 1:i + block_size + 1]
        for i in starts.tolist()
    ])

    return x.to(device), y.to(device)


# ============================================================
# STEP 6: SELF-ATTENTION HEAD
# ============================================================

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
        _, T, _ = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1)
        wei = wei * (k.shape[-1] ** -0.5)

        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        return wei @ v


# ============================================================
# STEP 7: MULTI-HEAD ATTENTION
# ============================================================

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([
            Head(head_size)
            for _ in range(num_heads)
        ])

        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat(
            [head(x) for head in self.heads],
            dim=-1
        )

        return self.dropout(self.proj(out))


# ============================================================
# STEP 8: FEED-FORWARD NETWORK
# ============================================================

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


# ============================================================
# STEP 9: TRANSFORMER BLOCK
# ============================================================

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


# ============================================================
# STEP 10: GPT MODEL
# ============================================================

class GPT(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(
            vocab_size,
            n_embd
        )

        self.position_embedding_table = nn.Embedding(
            block_size,
            n_embd
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
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(self, idx, targets=None):
        _, T = idx.shape

        tok_emb = self.token_embedding_table(idx)

        positions = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(positions)

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)
        loss = None

        if targets is not None:
            _, _, C = logits.shape

            logits_flat = logits.reshape(-1, C)
            targets_flat = targets.reshape(-1)

            loss = F.cross_entropy(
                logits_flat,
                targets_flat
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(
                probs,
                num_samples=1
            )

            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# ============================================================
# STEP 11: CREATE MODEL
# ============================================================

model = GPT().to(device)

number_of_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print("\nModel parameters:", number_of_parameters)


# ============================================================
# STEP 12: OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


# ============================================================
# STEP 13: LOSS ESTIMATION
# ============================================================

@torch.no_grad()
def estimate_loss():
    model.eval()
    results = {}

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters, device=device)

        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss

        results[split] = losses.mean().item()

    model.train()

    return results


# ============================================================
# STEP 14: TRAINING
# ============================================================

print("\n")
print("=" * 60)
print("STARTING EXPERIMENT TRAINING")
print("=" * 60)

for iteration in range(max_iters):

    if iteration % eval_interval == 0 or iteration == max_iters - 1:
        losses = estimate_loss()

        print(
            f"Step {iteration:4d} | "
            f"Train loss: {losses['train']:.4f} | "
            f"Validation loss: {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")

    _, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# ============================================================
# STEP 15: SAVE EXPERIMENT CHECKPOINT
# Never overwrite my_llm.pth.
# ============================================================

checkpoint = {
    "model_state_dict": model.state_dict(),
    "stoi": stoi,
    "itos": itos,
    "config": {
        "vocab_size": vocab_size,
        "block_size": block_size,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "dropout": dropout
    },
    "experiment": {
        "dataset_file": str(DATA_PATH),
        "dataset_characters": len(text),
        "training_characters": len(train_data),
        "validation_characters": len(val_data),
        "max_iters": max_iters,
        "learning_rate": learning_rate
    }
}

# Refuse to overwrite an existing experiment checkpoint.
if OUTPUT_PATH.exists():
    raise SystemExit(
        f"ERROR: {OUTPUT_PATH} already exists. "
        "Rename or back it up before running another experiment."
    )

torch.save(checkpoint, OUTPUT_PATH)

print("\nExperiment model saved as:", OUTPUT_PATH)


# ============================================================
# STEP 16: SAMPLE GENERATION
# Generate a short sample from the trained model.
# ============================================================

print("\n")
print("=" * 60)
print("EXPERIMENT GENERATED TEXT")
print("=" * 60)

model.eval()

start_text = text[0]

context = torch.tensor(
    [encode(start_text)],
    dtype=torch.long,
    device=device
)

generated_ids = model.generate(
    context,
    max_new_tokens=500
)

print(decode(generated_ids[0].tolist()))

print("\n")
print("=" * 60)
print("NESAMANI LLM EXPERIMENT FINISHED")
print("=" * 60)
