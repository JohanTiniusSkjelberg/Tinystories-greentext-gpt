import json
import torch
from model import BigramLanguageModel, GPTConfig
import tiktoken
from train import estimate_loss, training_run, load_tokens

# Hyper
max_iters = 4000
eval_interval = 100
eval_iters = 50
batch_size = 32

with open("data/greentext_synth.txt", "r", encoding="utf-8") as f:
    greentext_synth = f.read()
with open("data/greentext_darwin_good.txt", "r", encoding="utf-8") as f:
    greentext_darwin = f.read()

greentext = greentext_darwin + greentext_synth

device = 'cuda' if torch.cuda.is_available() else 'cpu'

ckpt = torch.load('tinystories.pt', map_location=device)

enc = tiktoken.get_encoding("gpt2")
config = GPTConfig(**ckpt['config'])
model = BigramLanguageModel(config)
model.load_state_dict(ckpt['state_dict'])

green_tensor = torch.tensor(enc.encode(greentext, allowed_special={'<|endoftext|>'}), dtype=torch.long)
n = int(0.9 * len(green_tensor))
green_train = green_tensor[:n]
green_val = green_tensor[n:]

# TinyStories val set: shows how much of the base task the model forgets
tiny_val = load_tokens("data/TinyStories-valid.txt", enc)

model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

train_losses, history = training_run(model, optimizer, max_iters, green_train, green_val, eval_interval, eval_iters,
            batch_size , ckpt_path='green_tinystories.pt', encoding=enc.name, extra_val={'tiny_val': tiny_val})

# One record for each eval step: step, train, val (greentext), tiny_val (TinyStories)
with open('finetune_losses.json', 'w') as f:
    json.dump(history, f, indent=1)

losses = estimate_loss(model, green_train, green_val, eval_iters, batch_size)

print(f"step {max_iters}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

prompt = "> "
idx = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)
print(enc.decode(model.generate(idx, 300)[0].tolist()))

