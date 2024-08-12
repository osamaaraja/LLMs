import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        weights = q @ k.transpose(-2, -1) * C**-0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)

        v = self.value(x)
        out = weights @  v
        return out

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # a table of size vocab_size x vocab_size
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx) # batch, context length, vocab size
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = token_emb + pos_emb
        x = self.sa_head(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            batch, context, vocab = logits.shape
            logits = logits.view(-1, vocab)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets) # assessing the quality of the logits with respect to targets
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i + block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x,y)
            losses[k] += loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# hyperparameters
block_size = 8
batch_size = 16
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
eval_iters = 200
n_embd = 32
device ='cuda' if torch.cuda.is_available() else 'cpu'


with open("../books/JungleBook.txt", 'r', encoding=u'utf-8') as file:
    text = file.read()

# all unique characters in the data
chars = sorted(list(set(text))) # getting all the characters in the dataset
vocab_size = len(chars)

# mapping the characters to integers
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encoder: string -> list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: list of integers -> string

data = torch.tensor(encode(text), dtype=torch.long)

# splitting the data into train and val
n = int(0.9*len(data))
train_data = data[:n] # 90% of the data
val_data = data[n:] # 10% of the data

model = BigramLanguageModel()
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# training loop
for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'step: {iter}, train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}')

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=300)[0].tolist()))
