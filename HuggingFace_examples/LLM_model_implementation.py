from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("DistilBert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-cased", num_labels=2)
# In the model above, since we are trying to perform clasification, the number of labels is 2

input_str = "Hugging Face Transformers is great!"

model_inputs = tokenizer(input_str, return_tensors="pt")

# Option 1
model_outputs = model(input_ids=model_inputs.input_ids, attention_mask=model_inputs.attention_mask)

# Option 2
model_outputs = model(**model_inputs) # the star star unpacks the dictionary of key and values
print(f"model_inputs: {model_inputs}\nmodel_outputs: {model_outputs}")
print(f"Distribution over the labels: {torch.softmax(model_outputs.logits, dim=-1)}") # logits are the raw output scores from the model

# the models from the Hugging Face are just Pytorch Modules.
# Training

label = torch.tensor([1]) # setting a label 1
loss = torch.nn.functional.cross_entropy(model_outputs.logits, label)
print(f"loss: {loss}")
loss.backward()

# for getting the parameters
print(f"The list of parameters:\n{list(model.named_parameters())[0]}")

########################################################################################################################
# Hugging Face provides an additional easy way for calculating the loss without using the torch modules
# model_inputs = tokenizer(input_str, return_tensors="pt")
# labels = ["NEGATIVE", "POSITIVE"]
# model_inputs['labels'] = torch.tensor([1])
# model_outputs = model(**model_inputs)
# print(model_outputs)
# print(f"Model Predictions: {labels[model_outputs.logits.argmax()]}")
########################################################################################################################

