'''
Sentiment Analysis using Pretrained Transformer from Hugging Face
https://huggingface.co/siebert/sentiment-roberta-large-english
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
    print("-----------------------------------------------------------------------")
    print("Decode:   ", decoded_str)
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
