def prune_linear(model, all_linear_data, threshold):
  pruning_map = all_linear_data.query(f'SmallestAverageMagnitude <= {threshold}')
  pruning_map = pruning_map.assign(PruneBias = pruning_map['AverageMagnitude'] <= threshold)
  target_layers = pruning_map['Layer'].unique()
  for name, m in model.named_modules():
    if isinstance(m, nn.Linear) and (name in target_layers):
      df = pruning_map.loc[pruning_map['Layer'] == name]
      for _, row in df.iterrows():
        neuron_type = row['NeuronType']
        col_index = row['ColumnIndex']

        # Prune the weights
        if neuron_type == 'linear_input':
            m.weight.data[:, col_index] = 0  # Zero out the corresponding row
        elif neuron_type == 'linear_output':
            m.weight.data[col_index, :] = 0  # Zero out the corresponding column
            if row['PruneBias']:
                m.bias.data[col_index] = 0

  # Return the pruned model
  return model

def prune_activations(model, all_activation_data, threshold, mapping):
  # for opt model 1.3 mapping=['activation_fn', 'fc1']
  # for phi-1.3 mapping=['act', 'fc1']
  pruning_map = all_activation_data.query(f'AverageMagnitude <= {threshold}')
  target_layers = [s.replace(mapping[0], mapping[1]) for s in pruning_map['Layer'].unique()]
  for name, m in model.named_modules():
    if isinstance(m, nn.Linear) and (name in target_layers):
      df = pruning_map.loc[pruning_map['Layer'] == name.replace(mapping[1],mapping[0])]
      for _, row in df.iterrows():
        if row['AverageMagnitude'] <= threshold:
          m.weight.data[row['ColumnIndex'], :] = 0  # Zero out the corresponding column of weights
          m.bias.data[row['ColumnIndex']] = 0       # Zero out the corresponding column in bias
  return model

def prune_embedding(model, normalized_counts, special_tokens, threshold, names):
  # Note this will slightly change perplexity as lm_head is automatically handled
  normalized_counts_s = normalized_counts
  normalized_counts_s[list(special_tokens)] = np.inf
  pruning_map = torch.where(normalized_counts_s <= threshold)[0]
  for name, m in model.named_modules():
    if isinstance(m, nn.Embedding) and (name in names):
      m.weight.data[pruning_map, :] = 0
    if isinstance(m, nn.Linear) and (name in names):
      m.weight.data[pruning_map, :] = 0
  return model
