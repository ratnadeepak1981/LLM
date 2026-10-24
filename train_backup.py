# ============================================================
# NESAMANI LLM — COMPLETE TINY GPT LANGUAGE MODEL
# File: train.py
# Framework: Python + PyTorch
# Architecture: Character-level GPT / Transformer
# ============================================================


# ============================================================
# STEP 1: CONFIGURATION
# Define the model architecture and training settings.
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

# Training configuration
batch_size = 32
block_size = 128
max_iters = 3000
eval_interval = 300
learning_rate = 3e-4
eval_iters = 50

# GPT architecture configuration
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.1

# Select GPU when CUDA is available; otherwise use CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("NESAMANI LLM")
print("=" * 60)
print("Device:", device)


# ============================================================
# STEP 2: LOAD DATA
# Read the training text from train.txt.
# ============================================================

try:
    with open("train.txt", "r", encoding="utf-8") as f:
        text = f.read()

except FileNotFoundError:
    raise SystemExit(
        "ERROR: train.txt was not found. "
        "Place it in the same folder as train.py."
    )

print("Dataset characters:", len(text))


# ============================================================
# STEP 3: CHECK DATASET
# Ensure the dataset is large enough for training sequences.
# ============================================================

minimum_required = block_size + 2

if len(text) < minimum_required:
    raise SystemExit(
        f"ERROR: Dataset is too small. "
        f"At least {minimum_required} characters are required."
    )


# ============================================================
# STEP 4: CREATE VOCABULARY
# Assign an integer ID to each unique character.
# ============================================================

chars = sorted(set(text))
vocab_size = len(chars)

# Character-to-number mapping.
stoi = {ch: i for i, ch in enumerate(chars)}

# Number-to-character mapping.
itos = {i: ch for i, ch in enumerate(chars)}

print("Vocabulary size:", vocab_size)


# ============================================================
# STEP 5: ENCODER / DECODER
# Convert text into token IDs and IDs back into text.
# ============================================================

def encode(s):
    """Convert a string into a list of character IDs."""
    return [stoi[c] for c in s]


def decode(ids):
    """Convert character IDs back into a string."""
    return "".join(itos[int(i)] for i in ids)


# ============================================================
# STEP 6: CONVERT TEXT TO TENSOR
# PyTorch uses integer token IDs for embedding lookup.
# ============================================================

data = torch.tensor(encode(text), dtype=torch.long)


# ============================================================
# STEP 7: TRAIN / VALIDATION SPLIT
# Reserve 90% for training and 10% for validation.
# ============================================================

split_index = int(0.9 * len(data))

train_data = data[:split_index]
val_data = data[split_index:]

if len(train_data) <= block_size:
    raise SystemExit("ERROR: Training data is too small.")

if len(val_data) <= block_size:
    raise SystemExit("ERROR: Validation data is too small.")

print("Training characters:", len(train_data))
print("Validation characters:", len(val_data))


# ============================================================
# STEP 8: BATCH FUNCTION
# Build input sequences and targets shifted by one token.
# ============================================================

def get_batch(split):
    """Return a batch of inputs and next-token targets."""

    if split == "train":
        data_source = train_data
    else:
        data_source = val_data

    # Select random starting positions.
    max_start = len(data_source) - block_size - 1

    starts = torch.randint(
        0,
        max_start + 1,
        (batch_size,)
    )

    # x contains input sequences.
    # y contains the same sequences shifted one token ahead.
    x = torch.stack([
        data_source[i:i + block_size]
        for i in starts.tolist()
    ])

    y = torch.stack([
        data_source[i + 1:i + block_size + 1]
        for i in starts.tolist()
    ])

    # Move both tensors to the selected device.
    x = x.to(device)
    y = y.to(device)

    return x, y


# ============================================================
# STEP 9: LOSS ESTIMATION
# Measure average prediction loss on training and validation.
# ============================================================

@torch.no_grad()
def estimate_loss():
    """Evaluate the model without calculating gradients."""

    model.eval()
    results = {}

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters, device=device)

        for k in range(eval_iters):
            x, y = get_batch(split)

            # Forward pass returns predictions and loss.
            logits, loss = model(x, y)

            losses[k] = loss

        results[split] = losses.mean().item()

    # Restore training mode after evaluation.
    model.train()

    return results


# ============================================================
# STEP 10: SELF-ATTENTION HEAD
# Allow each token to attend to earlier tokens.
# ============================================================

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()

        # Project input embeddings into query, key and value vectors.
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Lower-triangular mask prevents looking at future tokens.
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        B, T, C = x.shape

        # Calculate query, key and value representations.
        k = self.key(x)
        q = self.query(x)

        # Calculate scaled attention scores.
        wei = q @ k.transpose(-2, -1)
        wei = wei * (k.shape[-1] ** -0.5)

        # Hide future positions.
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        # Convert scores into attention probabilities.
        wei = F.softmax(wei, dim=-1)

        # Apply dropout to attention weights.
        wei = self.dropout(wei)

        # Combine the values using attention weights.
        v = self.value(x)
        out = wei @ v

        return out


# ============================================================
# STEP 11: MULTI-HEAD ATTENTION
# Run several attention heads in parallel.
# ============================================================

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()

        # Each head learns a different attention representation.
        self.heads = nn.ModuleList([
            Head(head_size)
            for _ in range(num_heads)
        ])

        # Project concatenated heads back to embedding size.
        self.proj = nn.Linear(
            num_heads * head_size,
            n_embd
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # Concatenate all attention-head outputs.
        out = torch.cat(
            [head(x) for head in self.heads],
            dim=-1
        )

        # Mix information from all heads.
        out = self.proj(out)

        # Regularization during training.
        out = self.dropout(out)

        return out


# ============================================================
# STEP 12: FEED-FORWARD NETWORK
# Process each token representation with neural-network layers.
# ============================================================

class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# STEP 13: TRANSFORMER BLOCK
# Combine attention, feed-forward processing and residual paths.
# ============================================================

class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()

        # Each attention head receives part of the embedding.
        head_size = n_embd // n_head

        # Transformer sublayers.
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)

        # Layer normalization stabilizes training.
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):

        # Attention with a residual connection.
        x = x + self.sa(self.ln1(x))

        # Feed-forward network with a residual connection.
        x = x + self.ffwd(self.ln2(x))

        return x


# ============================================================
# STEP 14: GPT MODEL
# Assemble embeddings, Transformer blocks and output layer.
# ============================================================

class GPT(nn.Module):

    def __init__(self):
        super().__init__()

        # Convert token IDs into learned vectors.
        self.token_embedding_table = nn.Embedding(
            vocab_size,
            n_embd
        )

        # Represent the position of each token.
        self.position_embedding_table = nn.Embedding(
            block_size,
            n_embd
        )

        # Stack multiple Transformer blocks.
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head)
            for _ in range(n_layer)
        ])

        # Final normalization and vocabulary prediction layer.
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # STEP 15: WEIGHT INITIALIZATION
        # Initialize model weights before training.
        self.apply(self._init_weights)

    # ========================================================
    # STEP 15: WEIGHT INITIALIZATION
    # Give neural-network layers suitable starting weights.
    # ========================================================

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

    # ========================================================
    # STEP 16: FORWARD PASS
    # Predict the next character and optionally calculate loss.
    # ========================================================

    def forward(self, idx, targets=None):

        B, T = idx.shape

        # STEP 16A: Token embeddings.
        tok_emb = self.token_embedding_table(idx)

        # STEP 16B: Position embeddings.
        positions = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(positions)

        # STEP 16C: Combine token and position information.
        x = tok_emb + pos_emb

        # STEP 16D: Process through Transformer blocks.
        x = self.blocks(x)

        # STEP 16E: Normalize the final representation.
        x = self.ln_f(x)

        # STEP 16F: Produce a score for every vocabulary token.
        logits = self.lm_head(x)

        loss = None

        # Calculate cross-entropy loss when targets are supplied.
        if targets is not None:

            B, T, C = logits.shape

            # Flatten predictions and targets for cross-entropy.
            logits_flat = logits.reshape(B * T, C)
            targets_flat = targets.reshape(B * T)

            loss = F.cross_entropy(
                logits_flat,
                targets_flat
            )

        return logits, loss

    # ========================================================
    # STEP 17: GENERATE TEXT
    # Predict and append one new character at a time.
    # ========================================================

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):

            # Keep only the most recent context window.
            idx_cond = idx[:, -block_size:]

            # Predict the next token.
            logits, loss = self(idx_cond)

            # Use predictions at the final position.
            logits = logits[:, -1, :]

            # Convert scores into probabilities.
            probs = F.softmax(logits, dim=-1)

            # Sample the next character.
            idx_next = torch.multinomial(
                probs,
                num_samples=1
            )

            # Append the generated token.
            idx = torch.cat(
                (idx, idx_next),
                dim=1
            )

        return idx


# ============================================================
# STEP 18: CREATE MODEL
# Instantiate the GPT model and count its parameters.
# ============================================================

model = GPT().to(device)

number_of_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print("Model parameters:", number_of_parameters)


# ============================================================
# STEP 19: OPTIMIZER
# Configure AdamW to update model parameters.
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


# ============================================================
# STEP 20: TRAINING
# Repeat batches, calculate loss and update model weights.
# ============================================================

print("\n")
print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)

for iteration in range(max_iters):

    # Periodically evaluate training and validation loss.
    if iteration % eval_interval == 0 or iteration == max_iters - 1:

        losses = estimate_loss()

        print(
            f"Step {iteration:4d} | "
            f"Train loss: {losses['train']:.4f} | "
            f"Validation loss: {losses['val']:.4f}"
        )

    # Obtain a training batch.
    xb, yb = get_batch("train")

    # Forward pass calculates predictions and loss.
    logits, loss = model(xb, yb)

    # Clear gradients from the previous iteration.
    optimizer.zero_grad(set_to_none=True)

    # Backpropagation calculates gradients.
    loss.backward()

    # Update model parameters.
    optimizer.step()


# ============================================================
# STEP 21: TRAINING FINISHED
# Report that the training loop has completed.
# ============================================================

print("\n")
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)


# ============================================================
# STEP 22: SAVE MODEL
# Save the trained parameters, vocabulary and configuration.
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
    }
}

torch.save(checkpoint, "my_llm.pth")

print("\nModel saved as: my_llm.pth")


# ============================================================
# STEP 23: GENERATE TEXT
# Start with the first character and generate a continuation.
# ============================================================

print("\n")
print("=" * 60)
print("GENERATED TEXT")
print("=" * 60)

# Use the first dataset character as the initial context.
start_text = text[0]

# Encode the starting character and move it to the device.
context = torch.tensor(
    [encode(start_text)],
    dtype=torch.long,
    device=device
)

# Switch to evaluation mode for text generation.
model.eval()

# Generate a continuation of 500 characters.
generated_ids = model.generate(
    context,
    max_new_tokens=500
)

# Decode the generated token IDs into readable text.
generated_text = decode(generated_ids[0].tolist())

print(generated_text)

print("\n")
print("=" * 60)
print("NESAMANI LLM FINISHED")
print("=" * 60)

# End of the 23-step tiny GPT training pipeline.
