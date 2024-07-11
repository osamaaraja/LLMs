# datasets are available on Hugging face

from datasets import load_dataset, DatasetDict # the DatasetDict class is from Hugging face
from transformers import AutoModel, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("DistilBert-base-cased")

imdb_dataset = load_dataset('imdb')

# taking the first 50 tokens for quick run
def truncate(example):
    return {'text': " ".join(example['text'].split()[:10]),
            'label': example['label']}

# taking 128 random examples for training and 32 examples for validation
small_imdb_dataset = DatasetDict(
    train = imdb_dataset['train'].shuffle(seed=111).select(range(128)).map(truncate),
    val = imdb_dataset['train'].shuffle(seed=111).select(range(128, 160)).map(truncate),
)

print(f'small_imdb_dataset: {small_imdb_dataset}')

print('Train',small_imdb_dataset['train'][:10]) # looking at the first 10 entries of the training dataset


# preparing the data into batches and then tokenizing the dataset into a batches of 16

small_tokenized_dataset = small_imdb_dataset.map(
    lambda example:tokenizer(example['text'], padding=True, truncation=True)
)

small_tokenized_dataset = small_tokenized_dataset.remove_columns(['text'])
small_tokenized_dataset = small_tokenized_dataset.rename_columns({'label':'label'})
small_tokenized_dataset.set_format("torch") # setting it to torch so it can be passed to the model

print(f'small_tokenized_dataset: {small_tokenized_dataset['train'][:2]}')

# now for training, need to use the Pytorch Dataloader

#for training a BERT model (or any transformer model from the Hugging Face transformers library), the text data should always be tokenized.
# Tokenization is a crucial preprocessing step that converts raw text into a format that the model can understand and process.

from torch.utils.data import DataLoader

train_dataloader = DataLoader(small_tokenized_dataset['train'], batch_size=16)
val_dataloader = DataLoader(small_tokenized_dataset['val'], batch_size=16)

from transformers import AdamW, get_linear_schedule_with_warmup # Adam with weight decay and linear scheduler for learning rate
from tqdm import tqdm

model = AutoModel.from_pretrained("distilbert-base-cased",num_labels=2)

num_epochs = 3
num_training_steps = 3 + len(train_dataloader)
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)


best_val_loss = float('inf')
loss = 0
progress_bar = tqdm(range(num_training_steps))


for epoch in range(num_epochs):
    model.train()
    for batch_i, batch in enumerate(train_dataloader):

        outputs = model(**batch)
        optimizer.zero_grad()
        outputs.loss.backward()
        optimizer.step()
        lr_scheduler.step()
        progress_bar.update(1)

    model.eval()
    for batch_i, batch in enumerate(val_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        loss += outputs.loss

    avg_val_loss = loss / len(val_dataloader)
    print(f'Validation Loss:  {avg_val_loss}')
    if avg_val_loss < best_val_loss:
        print('Saving checkpoint...')
        best_val_loss = avg_val_loss
        torch.save({
            "epoch":epoch,
            "mode_state_dict": model.state_dict(),
            "optimizer_state_dict":optimizer.state_dict(),
            "val_loss":best_val_loss
        },
            f"epoch_{epoch}.pt"
        )











