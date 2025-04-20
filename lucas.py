import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200

torch.manual_seed(1337)

with open("shakespeare.txt", "r") as file:
    text = file.read()

chars = sorted(list(set(text)))
vocab = len(chars)
print("".join(chars))
print(vocab)


stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

n = (int)(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


block_size = 8
train_data[:block_size + 1]

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # finds 4 random starting indices for getting blocks
    x = torch.stack([data[i:i+block_size] for i in ix]) # gets a stack of the 4 blocks of 8 values
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # gets the next value (which is what we're trying to predict)
    return x, y

xb, yb = get_batch("train")


# bigram is a pair of consecutive words in a sequence
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, inputs, targets=None):
        logits = self.token_embedding_table(inputs) # (Batch, Time, Channel)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, inputs, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(inputs)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            inputs_next = torch.multinomial(probs, num_samples=1) # (B, 1) as there is one prediction for what comes next
            inputs = torch.cat((inputs, inputs_next), dim=1)
        
        return inputs

m = BigramLanguageModel(vocab)
logits, loss = m(xb, yb) # passing inputs and targets
print(logits.shape)
print(loss)

inputs = torch.zeros((1, 1), dtype=torch.long)
print(decode(m.generate(inputs, max_new_tokens=100)[0].tolist()))

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for steps in range(10000):
    xb, yb = get_batch("train")

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


inputs = torch.zeros((1, 1), dtype=torch.long)
print(decode(m.generate(inputs, max_new_tokens=1000)[0].tolist()))





