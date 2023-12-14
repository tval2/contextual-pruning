"""
Example contextual-pruning using Phi-1.5 model and wikitext-2 dataset (+ fine-tuning)
"""
import numpy as np
import tqdm
import json
import torch
import random
from torch import nn
import transformers as tfmr
from datasets import load_dataset, inspect_dataset, load_dataset_builder, Split, concatenate_datasets, Dataset
from functools import partial
import gc
import pandas as pd
import matplotlib.pyplot as plt
import bitsandbytes as bnb
from torch.utils.data import DataLoader
from torch.cuda import empty_cache

#Let's first evaluate the perplexity and model size.

USE_8BIT = True
model_path = "microsoft/phi-1_5" #"facebook/opt-1.3b" #
tokenizer = tfmr.AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = tfmr.AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, load_in_8bit=USE_8BIT).eval()

dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
dataset = dataset.shuffle(seed=42)
testenc = tokenizer("\n\n".join(dataset['text']), return_tensors='pt')
testenc = testenc.input_ids.to(model.device)

# Evaluate the model
print('Before Pruning')
model_perplexity = show_model_stats(model, testenc, set_basline=True)

# Take a peak at what a decoder layer in the transformer looks like
print(model)

#Flatten the neuron dictionary
all_dict = dict()
neuron_dicts = get_calib_feat(model, testenc, all_dict)
neuron_type = 'activations'
activation_df = flatten_neuron_dict(neuron_dicts[neuron_type], neuron_type=neuron_type)
neuron_type = 'linear_input'
linear_input_df = flatten_neuron_dict(neuron_dicts[neuron_type], neuron_type=neuron_type)
neuron_type = 'linear_output'
linear_output_df = flatten_neuron_dict(neuron_dicts[neuron_type], neuron_type=neuron_type)
all_neuron_df = pd.concat([linear_input_df, linear_output_df, activation_df], axis=0)
all_neuron_df = all_neuron_df.assign(SmallestAverageMagnitude = all_neuron_df[["AverageMagnitude", "UnbiasedAverageMagnitude"]].min(axis=1))
print(all_neuron_df.shape)
assert all_neuron_df['Count'].nunique() == 1
all_neuron_df.head(3)

#Counts
special_tokens = get_special_tokens(tokenizer)
token_counts = count_all_tokens(testenc, tokenizer)
normalized_counts = token_counts / token_counts.sum()

#Linear Pruning
pruned_model = prune_linear(model, all_neuron_df.query(f"NeuronType in {['linear_input','linear_output']}"), threshold=0)

#Evaluate the linear-pruned model
print('After Linear Pruning')
model_perplexity = show_model_stats(pruned_model, testenc)

#Activations Pruning
pruned_model = prune_activations(pruned_model, all_neuron_df.query(f"NeuronType in {['activations']}"), threshold=0, mapping=['act', 'fc1'])

#Evaluate the activations-pruned model
print('After Activation Pruning')
model_perplexity = show_model_stats(pruned_model, testenc)

#Embeddings Pruning
pruned_model = prune_embedding(pruned_model, normalized_counts, special_tokens, threshold=0, names = ['transformer.embd.wte', 'lm_head.linear'])

#Evaluate the embedding-pruned model
print('After Embedding Pruning')
model_perplexity = show_model_stats(pruned_model, testenc)

#reset dataset for fine-tuning
dataset_ft = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
dataset_ft = dataset_ft.shuffle(seed=42)
tokenizer.pad_token = ' '
tokenized_inputs = tokenizer("\n\n".join(dataset_ft['text']), return_tensors="pt", truncation=True, max_length=512, padding="max_length")
tokenized_dataset = LLMTokenizedDataset(tokenized_inputs)
dataloader = DataLoader(tokenized_dataset, batch_size=2048, shuffle=True)

# Set train parameters based on model precision
num_finetune_epochs = 260 if USE_8BIT else 5  # Epochs to recover roughly accuracy
optimizer_type = bnb.optim.AdamW8bit if USE_8BIT else torch.optim.AdamW
dtype = None if USE_8BIT else torch.bfloat16  # .to(dtype) does not work for load_in_8bit models

#Fine-tune Model
fine_tune_model = pruned_model
optimizer = optimizer_type(fine_tune_model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_finetune_epochs)
criterion = nn.CrossEntropyLoss()
pruning_mask = create_pruning_mask(fine_tune_model)
for epoch in range(num_finetune_epochs):
    empty_cache()
    avg_loss = train(fine_tune_model, dataloader, optimizer, scheduler, pruning_mask, dtype=dtype)
    perplexity = evaluate_ft(fine_tune_model, dataloader, dtype=dtype)
    print(f'Epoch {epoch+1}: Avg Loss {avg_loss:.2f}, Perplexity {perplexity:.2f}')

#Evaluate the fine-tuned pruned model
print('After Fine Tuning')
model_perplexity = show_model_stats(fine_tune_model, testenc)
