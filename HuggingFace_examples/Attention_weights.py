from transformers import AutoModel, AutoTokenizer
import torch
import matplotlib.pyplot as plt

tokenizer = AutoTokenizer.from_pretrained("DistilBert-base-cased")
model = AutoModel.from_pretrained("distilbert-base-cased",output_attentions=True, output_hidden_states=True)
model.eval() # since we are not training so no need to calculate the gradients thus setting the model in eval model

input_str = "Hugging Face Transformers is great!"

model_inputs = tokenizer(input_str, return_tensors="pt")
with torch.no_grad():
    model_outputs = model(**model_inputs)

print("Hidden state size (per layer):", model_outputs.hidden_states[0].shape)
print("Attention head size (per layer):", model_outputs.attentions[0].shape)

#print(model_outputs)

# looking at the representation per layer basis
tokens = tokenizer.convert_ids_to_tokens(model_inputs.input_ids[0])
print(f"tokens:\n{tokens}")

n_layers = len(model_outputs.attentions)
n_heads = len(model_outputs.attentions[0][0])
fig, axes = plt.subplots(6,12)
fig.set_size_inches(18.5*2, 10.5*2)
for layer in range(n_layers):
    for i in range(n_heads):
        axes[layer][i].imshow(model_outputs.attentions[layer][0,i])
        axes[layer][i].set_xticklabels(list(range(10)))
        axes[layer][i].set_xticklabels(labels=tokens, rotation="vertical")
        axes[layer][i].set_yticklabels(list(range(10)))
        axes[layer][i].set_yticklabels(labels=tokens)

        if layer == 5:
            axes[layer,i].set(xlabel=f"head={i}")
        if i == 0:
            axes[layer, i].set(ylabel=f"layer={layer}")

plt.subplots_adjust(wspace=0.3)
plt.show()

