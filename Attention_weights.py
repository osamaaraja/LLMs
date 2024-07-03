from transformers import AutoModel, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("DistilBert-base-cased")
model = AutoModel.from_pretrained("distilbert-base-cased",output_attentions=True, output_hidden_states=True)
model.eval() # since we are not training so no need to calculate the gradients thus setting the model in eval model

input_str = "Hugging Face Transformers is great!"

model_inputs = tokenizer(input_str, return_tensors="pt")
with torch.no_grad():
    model_outputs = model(**model_inputs)

print("Hidden state size (per layer):", model_outputs.hidden_states[0].shape)
print("Attention head size (per layer):", model_outputs.attentions[0].shape)

print(model_outputs)