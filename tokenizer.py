
class CharTokenizer:
    def __init__(self, chars):
        self.chars = list(chars)
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join([self.itos[i] for i in ids])

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))
        
