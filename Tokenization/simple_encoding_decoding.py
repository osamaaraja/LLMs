with open("../books/JungleBook.txt", 'r', encoding=u'utf-8') as file:
    text = file.read() # text is sequence of characters in python

print("length of the dataset in characters: ", len(text))

chars = sorted(list(set(text))) # getting all the characters in the dataset
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# the next step is to tokenize this text
# creating mapping from characters to integers
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s] # encoder: string -> list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: list of integers -> string

print(encode("hello world"))
print(decode(encode("hello world")))