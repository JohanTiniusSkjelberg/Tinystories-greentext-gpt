import torch
import torch.nn as nn
from torch.nn import functional as F

class GPTConfig:
    def __init__(self, vocab_size, block_size=256, n_embd=384, n_layer=6, n_head=6, dropout=0.2):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout

class Head(nn.Module):
    """ One head of self-attention """

    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B,T,T)
        wei = F.softmax(wei, dim=-1)  # (B,T,T)
        wei = self.dropout(wei)
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """

    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity """

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, config):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)  # final layer norm
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        token_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T,C)
        x = token_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last config.block_size tokens
            idx_cond = idx[:, -self.config.block_size:]  # crop context to config.block_size
            # Get the predictions
            logits, _ = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B,C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B,C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B,T+1)
        return idx

    def resize_vocab(self, new_vocab_size):
        # Add rows for new characters at the end. The old rows keep their values.
        old_vocab_size = self.config.vocab_size
        assert new_vocab_size >= old_vocab_size, "the vocabulary can only grow"
        old_emb = self.token_embedding_table
        old_head = self.lm_head
        device = old_emb.weight.device

        new_emb = nn.Embedding(new_vocab_size, self.config.n_embd).to(device)
        new_head = nn.Linear(self.config.n_embd, new_vocab_size).to(device)

        with torch.no_grad():
            # Copy the old rows
            new_emb.weight[:old_vocab_size] = old_emb.weight
            new_head.weight[:old_vocab_size] = old_head.weight
            new_head.bias[:old_vocab_size] = old_head.bias
            # New rows start as the mean of the old rows: an "average character"
            new_emb.weight[old_vocab_size:] = old_emb.weight.mean(dim=0)
            new_head.weight[old_vocab_size:] = old_head.weight.mean(dim=0)
            new_head.bias[old_vocab_size:] = old_head.bias.mean()

        self.token_embedding_table = new_emb
        self.lm_head = new_head
        self.config.vocab_size = new_vocab_size

