'''Only decoder based transformer implemented'''

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
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        weights = q @ k.transpose(-2, -1) * C**-0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        v = self.value(x)
        out = weights @  v
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self attention is parallel"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concatenating along the context length
        out = self.proj(out) # projection is the linear transformation of out
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd,),
                                 nn.ReLU(),
                                 nn.Linear(4 * n_embd, n_embd), # this is the projection layer going back into the residual pathway
                                 nn.Dropout(dropout)
                                 )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block"""
    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        # introducing layer normalization
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # x + skip connection
        x = x + self.ffwd(self.ln2(x)) # x + skip connection
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # a table of size vocab_size x vocab_size
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        #self.sa_head = MultiHeadAttention(4, n_embd//4) # 4 heads of 8-dimensional self-attention --> 4x8=32 which is n_embd
        #self.ffwd = FeedForward(n_embd)
        '''
        self.blocks = nn.Sequential(
            Block(n_embd, n_heads=4),
            Block(n_embd, n_heads=4),
            Block(n_embd, n_heads=4),
            nn.LayerNorm(n_embd)
        )
        '''
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx) # batch, context length, vocab size
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = token_emb + pos_emb
        #x = self.sa_head(x)
        #x = self.ffwd(x)
        x = self.blocks(x)
        x = self.ln_f(x)
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

# hyperparameters -> adjust as needed based on the capability of the GPU
block_size = 16
batch_size = 16
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 32
n_head = 4
n_layer = 4
dropout = 0.2
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
