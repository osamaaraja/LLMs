'''
Sentiment Analysis using Pretrained Transformer from Hugging Face
https://huggingface.co/siebert/sentiment-roberta-large-english

Additional info: GPT2 is decoder based model, BERT is encoder based model, BART and T5 are Encode and Decoder based models
'''

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import datasets
from collections import defaultdict, Counter
import json
from matplotlib import pyplot as plt
import numpy as np
import torch


# Defining a helper function to see what encoding is actually happening
def print_encoding(model_inputs, indent=4):
    indent_str = ' ' * indent
    print("{")
    for k, v in model_inputs.items():
        print(indent_str + k + ":")
        print(indent_str + indent_str + str(v))
    print("}")

def print_all(input, output, prediction, labels):
    print(f'Input: {input}')
    #print_encoding(input)
    print(f'Model Output: {output}')
    print(f'Prediction is: {labels[prediction]}')
    print()

def print_tokenization(input_str, input_tokens, input_ids, input_ids_special_tokens, decoded_str):
    print("Start:                 ", input_str)
    print("Tokenize:              ", input_tokens)
    print("Convert token to ids:  ", input_ids)
    print("Add special tokens:    ", input_ids_special_tokens)
    print("-"*100)
    print("Decode:   ", decoded_str)
    print()

def print_for_fast_tokenizer(input_str, inputs):
    print(input_str)
    print("-"*50)
    print(f'Number of tokens: {len(inputs)}')
    print(f"ids: {inputs.ids}")
    print(f"tokens: {inputs.tokens}")
    print(f"special_tokens: {inputs.special_tokens_mask}")
    print()
    char_idx = 8
    print(f"For example, the {char_idx + 1}th character of the string is '{input_str[char_idx]}',"+\
          f" and it's part of wordpiece {inputs.char_to_token(char_idx)}, '{inputs.tokens[inputs.char_to_token(char_idx)]}'")
    print()


# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("siebert/sentiment-roberta-large-english") # For the purpose of splitting text into tokens
# Initialize the model
model = AutoModelForSequenceClassification.from_pretrained("siebert/sentiment-roberta-large-english")

input = "I'm excited to learn about Hugging Face Transformers!"
tokenized_input = tokenizer(input, return_tensors="pt") # Tokenizing the inputs before passing them to the model
output = model(**tokenized_input)

labels = ["NEGATIVE", "POSITIVE"]
prediction = torch.argmax(output.logits)

print_all(input, output, prediction, labels)


# there are different kinds of tokenizers that can be used
# Autotokenizer is the most convinient one
from transformers import DistilBertTokenizer, DistilBertTokenizerFast
tokenizer_bert = DistilBertTokenizer.from_pretrained("DistilBert-base-cased")
tokenizer_fast = DistilBertTokenizerFast.from_pretrained("DistilBert-base-cased")

input_str = "Hugging Face Transformers are great!"

tokenized_input_auto = tokenizer(input_str)
tokenized_input_bert = tokenizer_bert(input_str)
tokenized_input_fast = tokenizer_fast(input_str)

print("AUTO:",tokenized_input_auto)
print("BERT:",tokenized_input_bert)
print("BERT FAST:",tokenized_input_fast)
print()

# getting some insights into the tokenization process
cls = [tokenizer.cls_token_id]
sep = [tokenizer.sep_token_id]

# tokenization steps
input_tokens = tokenizer.tokenize(input_str)
input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
input_ids_special_tokens = cls + input_ids + sep
decoded_str = tokenizer.decode(input_ids_special_tokens)

print_tokenization(input_str, input_tokens, input_ids, input_ids_special_tokens, decoded_str)

# for fast tokenizer the other option is as below
input = tokenizer._tokenizer.encode(input_str)
print_for_fast_tokenizer(input_str, input)

# The tokenizers can retunr Pytorch tensors
model_inputs = tokenizer("Hugging Face Tranformers is great!", return_tensors="pt")
print("Pytorch Tensor:")
print_encoding(model_inputs)

# adding mulitple strings into the tokenizer
model_inputs = tokenizer(["Hugging Face Transformers is great!", "The quick brown fox jumps over the laazy dog.",
                          "Then the dog got up and ran away because she didn't like foxes.",],
                         return_tensors="pt",
                         padding=True,
                         truncation=True
                         )
print(f"Pad token: {tokenizer.pad_token} , Pad token id: {tokenizer.pad_token_id}")
print("Padding:")
print_encoding(model_inputs)

# whole batch can be decoded at once
print(f"Batch Decode:\n {tokenizer.batch_decode(model_inputs.input_ids)}")
print(f"Batch Decode: (no special character)\n {tokenizer.batch_decode(model_inputs.input_ids, skip_special_tokens=True)}")




