import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')  # no window, only image files (for runs over ssh)
import matplotlib.pyplot as plt
from model import BigramLanguageModel, GPTConfig
import tiktoken

# config = GPTConfig(vocab_size=65, n_embd=32, n_layer=2, n_head=2, block_size=8)
# Model hyperparameters (used only to make GPTConfig)
block_size = 256
n_embd = 384
n_layer = 6
n_head = 6
dropout = 0.1

# Training hyperparameters
batch_size = 32 # maybe incresase to 64
# max_iters = 5000
max_iters = 20000
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval = 500
eval_iters = 50

torch.manual_seed(1337)

def save_checkpoint(model, path, encoding, **extra):
    ckpt = {
            'state_dict' : model.state_dict(),
            'config': dict(vars(model.config)),
            'encoding': encoding,
            **extra,
            }
    torch.save(ckpt, path)

def training_run(model, optimizer, max_iters, train_data, val_data, eval_interval, eval_iters,
                batch_size, ckpt_path=None, encoding='gpt2', extra_val=None):
    # If ckpt_path is given, the model with the lowest val loss is saved there
    # extra_val is a dict name -> data of more val sets to measure (not used for the checkpoint)
    # Returns the train loss of each step, and one dict of losses for each eval step
    train_losses = []
    history = []
    best_val = float('inf')
    for steps in range(max_iters):

        if steps % eval_interval == 0 or steps == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, eval_iters, batch_size)
            if ckpt_path is not None and losses['val'] < best_val:
                best_val = losses['val'].item()
                save_checkpoint(model, ckpt_path, encoding, step=steps, val_loss=best_val)

            record = {'step': steps, 'train': losses['train'].item(), 'val': losses['val'].item()}
            for name, data in (extra_val or {}).items():
                record[name] = eval_loss(model, data, eval_iters, batch_size).item()
            history.append(record)

            print(f"step {steps}: " + ", ".join(f"{k} loss {v:.4f}" for k, v in record.items() if k != 'step'))


        # Sample a batch of data
        xb, yb = get_batch(train_data, model.config.block_size, batch_size)

        # Evaluate the loss
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(xb, yb)

        train_losses.append(loss.item())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return train_losses, history

def get_batch(data, block_size, batch_size):
    # Generate a small batch of data of inputs x and targets y
    # data is a long tensor or a uint16 memmap from load_tokens; the model needs long
    ix = torch.randint(len(data) - block_size, (batch_size,)).tolist()
    x = torch.stack([torch.from_numpy(np.asarray(data[i:i+block_size], dtype=np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(np.asarray(data[i+1:i+block_size+1], dtype=np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def load_tokens(txt_path, enc, piece_chars=20_000_000):
    # Returns the tokens of txt_path as a uint16 memmap (the gpt2 vocabulary is less than 65536).
    # The first call encodes the text in pieces and saves the tokens to a .bin file.
    # Later calls only open that file.
    bin_path = os.path.splitext(txt_path)[0] + '.bin'
    if not os.path.exists(bin_path):
        print(f"encoding {txt_path} -> {bin_path}")
        tmp_path = bin_path + '.tmp'  # an interrupted run does not leave a part of a .bin file
        with open(txt_path, "r", encoding="utf-8") as f, open(tmp_path, "wb") as out:
            def write_piece(lines):
                ids = enc.encode(''.join(lines), allowed_special={'<|endoftext|>'})
                np.array(ids, dtype=np.uint16).tofile(out)

            piece, size = [], 0
            for line in f:
                # Cut only immediately before a separator: the tokens are then the same as from one encode call
                if size >= piece_chars and line.startswith('<|endoftext|>'):
                    write_piece(piece)
                    piece, size = [], 0
                piece.append(line)
                size += len(line)
            write_piece(piece)
        os.replace(tmp_path, bin_path)
    return np.memmap(bin_path, dtype=np.uint16, mode='r')

@torch.no_grad()
def eval_loss(model, data, eval_iters, batch_size):
    # Mean loss of eval_iters random batches of one data set
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(data, model.config.block_size, batch_size)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean()

def estimate_loss(model, train_data, val_data, eval_iters, batch_size):
    return {'train': eval_loss(model, train_data, eval_iters, batch_size),
            'val': eval_loss(model, val_data, eval_iters, batch_size)}

if __name__ == '__main__':
    enc = tiktoken.get_encoding("gpt2")

    config = GPTConfig(vocab_size=enc.n_vocab, block_size=block_size, n_embd=n_embd,
                       n_layer=n_layer, n_head=n_head, dropout=dropout)

    train_data = load_tokens("data/TinyStories-train.txt", enc)
    val_data = load_tokens("data/TinyStories-valid.txt", enc)
    print(f"train tokens {len(train_data)}, val tokens {len(val_data)}")

    # Split the data into train and validation sets 90% train, 10% validation
    # n = int(0.9 * len(data))
    # train_data = data[:n]
    # val_data = data[n:]

    model = BigramLanguageModel(config)
    m = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    losses, history = training_run(model, optimizer, max_iters, train_data, val_data, eval_interval, eval_iters,
                        batch_size ,ckpt_path='tinystories.pt', encoding=enc.name)

    plt.plot(losses, label='Loss 1')
    # plt.plot(losses2, label='Loss 2')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss over Iterations')
    plt.legend()
    plt.savefig('loss.png', dpi=150)
