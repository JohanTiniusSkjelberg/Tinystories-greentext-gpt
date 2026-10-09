import argparse
import torch
from torch.nn import functional as F
import tiktoken
from model import BigramLanguageModel, GPTConfig

parser = argparse.ArgumentParser(description="Generate text from a checkpoint made by train.py")
parser.add_argument('--ckpt', default='tinystories.pt')
parser.add_argument('--prompt', default='Once upon a time')
parser.add_argument('--max_new_tokens', type=int, default=300)
parser.add_argument('--temperature', type=float, default=0.8)  # lower = safer words, higher = more random
parser.add_argument('--top_k', type=int, default=50)  # sample only from the k best tokens, 0 = all tokens
parser.add_argument('--num_samples', type=int, default=1)
parser.add_argument('--seed', type=int, default=None)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.seed is not None:
    torch.manual_seed(args.seed)

ckpt = torch.load(args.ckpt, map_location=device)
enc = tiktoken.get_encoding(ckpt['encoding'])
config = GPTConfig(**ckpt['config'])
model = BigramLanguageModel(config)
model.load_state_dict(ckpt['state_dict'])
model.to(device)
model.eval()  # dropout off

# An empty prompt starts a new story from the separator token
start_ids = enc.encode(args.prompt, allowed_special={'<|endoftext|>'}) or [enc.eot_token]

with torch.no_grad():
    for _ in range(args.num_samples):
        idx = torch.tensor([start_ids], dtype=torch.long, device=device)
        new_ids = []
        for _ in range(args.max_new_tokens):
            idx_cond = idx[:, -config.block_size:]  # crop context to config.block_size
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / args.temperature  # (1,vocab_size)
            if args.top_k > 0:
                # All tokens below the k-th best get probability 0
                v, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (1,1)
            if idx_next.item() == enc.eot_token:  # the story is complete
                break
            idx = torch.cat((idx, idx_next), dim=1)
            new_ids.append(idx_next.item())
        # Decode all the new tokens together: one character can be more than one token
        print(args.prompt + enc.decode(new_ids))
        print('-' * 40)
