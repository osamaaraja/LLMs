import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import T

torch.manual_seed(42)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) # a table of size vocab_size x vocab_size

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # batch, context length, vocab size

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
            logits, loss = self(idx)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i + block_size+1] for i in ix])
    return x,y

with open("../books/JungleBook.txt", 'r', encoding=u'utf-8') as file:
    text = file.read() # text is sequence of characters in python

print("length of the dataset in characters: ", len(text))

chars = sorted(list(set(text))) # getting all the characters in the dataset
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# the next step is to tokenize this text
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s] # encoder: string -> list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: list of integers -> string

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)

# splitting the data into train and val
n = int(0.9*len(data))
train_data = data[:n] # 90% of the data
val_data = data[n:] # 10% of the data

# chunking the data for training
block_size = 8
batch_size = 16
epochs = 10000

model = BigramLanguageModel(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for steps in range(epochs):
    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    #print(f'Epoch {steps} ----> Loss: {loss.item()}')

print(decode(model.generate(idx=torch.zeros((1,1), dtype=torch.long), max_new_tokens=300)[0].tolist()))




