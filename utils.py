# From Lab 1
def get_model_sparsity(model: nn.Module) -> float:
    """
    calculate the sparsity of the given model
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    num_nonzeros, num_elements = 0, 0
    for param in model.parameters():
        num_nonzeros += param.count_nonzero()
        num_elements += param.numel()
    return 1 - float(num_nonzeros) / num_elements

# From Lab 1
def get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int:
    """
    calculate the total number of parameters of model
    :param count_nonzero_only: only count nonzero weights
    """
    num_counted_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_counted_elements += param.count_nonzero()
        else:
            num_counted_elements += param.numel()
    return num_counted_elements

# From Lab 4
def get_model_size(model: nn.Module, data_width=16, group_size=-1):

    if group_size != -1:
        data_width += (16 + 4) / group_size

    num_elements = 0
    for param in model.parameters():
        num_elements += param.numel()
    return num_elements * data_width

# From Lab 1
def get_model_size2(model: nn.Module, data_width=32, count_nonzero_only=False) -> int:
    """
    calculate the model size in bits
    :param data_width: #bits per element
    :param count_nonzero_only: only count nonzero weights
    """
    return get_num_parameters(model, count_nonzero_only) * data_width

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB

def get_true_model_size(model, count_nonzero_only=False, include_buffers=True, data_width=None):
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        el_num_bits = (param.element_size() if data_width is None else data_width) * Byte
        counter = param.count_nonzero if count_nonzero_only else param.numel
        param_size += counter() * el_num_bits

    if include_buffers:
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

    return param_size + buffer_size

# From Lab 4
def evaluate(model, testenc, batch_size=2048):
    nsamples = 40
    nlls = []
    for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * batch_size):((i + 1) * batch_size)].to(model.device)
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[:, (i * batch_size):((i + 1) * batch_size)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * batch_size
        nlls.append(neg_log_likelihood)
    if i == nsamples - 1:
        batch.to("cpu")
        del batch
        torch.cuda.empty_cache()

    return torch.exp(torch.stack(nlls).sum() / (nsamples * batch_size))

def get_calib_dataset(testenc, batch_size=2048):
    nsamples = 40
    samples = []
    for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * batch_size):((i + 1) * batch_size)].to(model.device)
        samples.append(batch)
    if i == nsamples - 1:
        batch.to("cpu")
        del batch
        torch.cuda.empty_cache()
    return samples

@torch.no_grad()
def get_calib_feat(model, testenc, full_dict=dict()):
    linear_input_neuron_dict, linear_output_neuron_dict = dict(), dict()
    def neuron_hook_linear(m, x, y, name):
        if isinstance(x, tuple):
          x = x[0]
        x = x.view(-1, x.shape[-1])
        y = y.view(-1, y.shape[-1])
        if m.bias is not None:
            z = y - m.bias if m.bias is not None else 0*y
            z = z.view(-1, z.shape[-1])
        if name not in linear_input_neuron_dict:
            linear_input_neuron_dict[name] = dict()
            linear_input_neuron_dict[name]['sum'] = x.abs().sum(dim=0).cpu().numpy()
            linear_input_neuron_dict[name]['count'] = x.shape[0]
            linear_output_neuron_dict[name] = dict()
            linear_output_neuron_dict[name]['sum'] = y.abs().sum(dim=0).cpu().numpy()
            linear_output_neuron_dict[name]['count'] = y.shape[0]
            if m.bias is not None:
              linear_output_neuron_dict[name]['unbiased_sum'] = z.abs().sum(dim=0).cpu().numpy()
        else:
            linear_input_neuron_dict[name]['sum'] += x.abs().sum(dim=0).cpu().numpy()
            linear_input_neuron_dict[name]['count'] += x.shape[0]
            linear_output_neuron_dict[name]['sum'] += y.abs().sum(dim=0).cpu().numpy()
            linear_output_neuron_dict[name]['count'] += y.shape[0]
            if m.bias is not None:
              linear_output_neuron_dict[name]['unbiased_sum'] += z.abs().sum(dim=0).cpu().numpy()

    act_dict = dict()
    def neuron_hook_activation(m, x, y, name):
        if y.dim() == 3:
          y = y[0]
        if name not in act_dict:
            act_dict[name] = dict()
            act_dict[name]['sum'] = y.abs().sum(dim=0).cpu().numpy()
            act_dict[name]['count'] = y.shape[0]
        else:
            act_dict[name]['sum'] += y.abs().sum(dim=0).cpu().numpy()
            act_dict[name]['count'] += y.shape[0]

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(partial(neuron_hook_linear, name=name)))
        elif isinstance(m, nn.ReLU) or isinstance(m, tfmr.activations.NewGELUActivation):
            hooks.append(m.register_forward_hook(partial(neuron_hook_activation, name=name)))

    print("Collecting neuron values...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = get_calib_dataset(testenc)
    pbar = tqdm.tqdm(samples)

    try:
      err = None
      for input_ids in pbar:
          input_ids = input_ids.to(device)
          model(input_ids)
    except Exception as e:
      err = e

    for hook in hooks:
        hook.remove()

    if err != None:
        raise(err)

    full_dict['linear_input'] = linear_input_neuron_dict
    full_dict['linear_output'] = linear_output_neuron_dict
    full_dict['activations'] = act_dict
    return full_dict

# This function just reorganizes the dictionary above into a dataframe
def flatten_neuron_dict(neuron_dict, neuron_type):
    # Create a list to hold flattened data
    flat_data = []

    # Iterate over each layer and neuron type in the dictionary
    for layer_name, data in neuron_dict.items():
      for col_idx, avg_val in enumerate(data['sum']/data['count']):
        # Append a record for each column (neuron) in the layer
        flat_data.append({
            'Layer': layer_name,
            'NeuronType': neuron_type,
            'ColumnIndex': col_idx,
            'AverageMagnitude': avg_val,
            'UnbiasedAverageMagnitude': data['unbiased_sum'][col_idx]/data['count'] if 'unbiased_sum' in data else None,
            'Count': data['count'],
        })

    # Convert to DataFrame
    df = pd.DataFrame(flat_data)
    return df

def plot_nonembed_neurons(neuron_df, one_curve=True):
  # Sort the DataFrame by 'AverageMagnitude' in descending order
  # Reset index to get a new column representing the sorted neuron indices
  col = 'SmallestAverageMagnitude'

  # Create the plot
  eps = np.finfo(float).eps
  plt.figure(figsize=(12, 6))
  if one_curve:
    sorted_all = neuron_df.sort_values(by=col, ascending=False)
    sorted_all.reset_index(drop=True, inplace=True)
    plt.plot(sorted_all[col].replace(0, eps), marker='o', markersize=2, linestyle='-', linewidth=1, label='All Neurons')
  else:
    sorted_input_linear = neuron_df.query("NeuronType == 'linear_input'").sort_values(by=col, ascending=False)
    sorted_input_linear.reset_index(drop=True, inplace=True)
    sorted_output_linear = neuron_df.query("NeuronType == 'linear_output'").sort_values(by=col, ascending=False)
    sorted_output_linear.reset_index(drop=True, inplace=True)
    sorted_activations = neuron_df.query("NeuronType == 'activations'").sort_values(by=col, ascending=False)
    sorted_activations.reset_index(drop=True, inplace=True)
    plt.plot(sorted_input_linear[col].replace(0, eps), marker='o', markersize=2, linestyle='-', linewidth=1, label='Input Linear Neurons')
    plt.plot(sorted_output_linear[col].replace(0, eps), marker='s', markersize=2, linestyle='-', linewidth=1, label='Output Linear Neurons')
    plt.plot(sorted_activations[col].replace(0, eps), marker='x', markersize=2, linestyle='-', linewidth=1, label='Activation Neurons')
  plt.yscale('log')
  plt.xscale('log')
  plt.title('Average Neuron Magnitudes of Neurons')
  plt.xlabel('Neuron Index (sorted)')
  plt.ylabel('Average Neuron Magnitude')
  plt.grid(True)
  plt.legend()
  plt.show()
  return plt

def get_special_tokens(tokenizer):
    """
    Get a list of special token IDs from the tokenizer.
    """
    special_tokens = set()
    # Add standard special tokens
    special_tokens.add(tokenizer.pad_token_id)
    special_tokens.add(tokenizer.bos_token_id)
    special_tokens.add(tokenizer.eos_token_id)
    special_tokens.add(tokenizer.unk_token_id)
    special_tokens.add(tokenizer.sep_token_id)
    special_tokens.add(tokenizer.cls_token_id)
    special_tokens.add(tokenizer.mask_token_id)
    special_tokens.add(tokenizer.pad_token_type_id)

    # Remove None values if any of the special tokens are not defined
    special_tokens.discard(None)

    return special_tokens

def count_all_tokens(data, tokenizer):
    max_token_id = max(tokenizer.get_vocab().values())
    token_counts = torch.zeros(max_token_id + 1, dtype=torch.long, device = data.device)
    flat_data = data.view(-1)
    unique_tokens, counts = torch.unique(flat_data, return_counts=True)
    token_counts.scatter_add_(0, unique_tokens, counts)
    return token_counts

def plot_token_frequencies(normalized_counts, special_tokens):
  sorted_counts, sorted_indices = torch.sort(normalized_counts, descending=True)
  plt.figure(figsize=(8, 6))
  plt.plot(sorted_counts.cpu().numpy(), label="Token Frequencies")
  plt.xlabel("Token (sorted)")
  plt.ylabel("Normalized Frequency")
  plt.xscale('log')
  plt.yscale('log')
  plt.title("Token Frequencies Sorted by Frequency Value")
  plt.legend()
  plt.show()
  print(f"Percent of tokens with freq of 0: {100*(normalized_counts <= 0).sum() / normalized_counts.shape[0]:0.2f}")
  return plt

BASELINE = 0
def show_model_stats(model, test_data=None, set_baseline=False):
    global BASELINE
    print(f'\nsparsity: {get_model_sparsity(model):.2f}')
    print(f'params: {get_num_parameters(model)/1000000:.2f} M')
    print(f'non-zero params: {get_num_parameters(model, count_nonzero_only=True)/1000000:.2f} M')
    print(f'32bit model size (only nonzero): {get_model_size2(model, count_nonzero_only=True)/ MiB:.2f} MiB')
    print(f"32bit model size: {get_model_size(model, data_width=32, group_size=128)/MiB:.2f} MiB")
    print(f"current model size (only nonzero): {get_true_model_size(model, count_nonzero_only=True)/MiB:.2f} MiB")
    print(f"current model size: {get_true_model_size(model)/MiB:.2f} MiB")
    model_perplexity = None
    if type(test_data) is not type(None):
        model_perplexity = evaluate(model, test_data)
        if set_baseline:
            BASELINE = model_perplexity
        print(f"base perplexity: {BASELINE:.3f}")
        print(f"\nmodel perplexity: {model_perplexity}")
    print(f"percent perplexity increase: {BASELINE / model_perplexity:.3f}")
    return model_perplexity

def ask(model, tokenizer, prompt, max_new=10):
    text_tokens = tokenizer(prompt, return_tensors='pt')
    inp = text_tokens.input_ids.to(model.device)
    with torch.no_grad():
        pred = model.generate(inp, max_new_tokens=max_new)
    return tokenizer.decode(pred.squeeze(0))

def construct_prompt(
        question, choices, context="", question_number=None,
        letter_choices=True, prefix="Answer the following question:\n", suffix=""):
    prompt = ""
    if context:
        prompt += context + "\n\n"
    prompt += prefix

    if question_number is not None:
        prompt += f"{question_number}) "

    choices_str = '\n'.join([
        f"{chr(ORD_a + i) if letter_choices else i + 1}. {choice_text}"
        for i, choice_text in enumerate(choices)
    ])
    prompt += question + '\n' + choices_str + suffix
    return prompt

def best_choice_by_perplexity(model, tokenizer, question, choices, letter_choices=True, question_choice_concat=' '):
    perplexities = []

    for choice in choices:
        text_tokens = tokenizer(question + question_choice_concat + choice.lstrip(' '), return_tensors='pt')
        inp = text_tokens.input_ids.to(model.device)
        with torch.no_grad():
            lm_logits = model(inp).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = inp[:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float()
        perplexities.append(torch.exp(neg_log_likelihood).item() / inp.shape[1])
    answer = np.argmin(perplexities) + 1
    if letter_choices:
        answer = chr(ORD_A + answer - 1)
    return answer

def model_question_eval(
        model, tokenizer, dataset, question_numbers=True, letter_choices=True,
        use_dataset_context=True, prompt_prefix="Answer the following question:\n",
        prompt_suffix="", answer_indicator="Answer: ", question_choice_concat=' ',
        max_num_questions=100, verbose=False):

    correct_count = 0
    confident_correct_count = 0
    best_guess_correct_count = 0
    num_questions = 0

    questions = dataset["questions"]
    context = dataset["context"] if use_dataset_context else ""
    for i, (question, answer_choice_dict) in enumerate(questions.items()):
        if i == max_num_questions:
            break
        question_number = i + 1 if question_numbers else None
        answer_choices = answer_choice_dict["choices"]
        # String of question followed by all choices
        prompt = construct_prompt(
            question, answer_choices,
            context=context, question_number=question_number,
            letter_choices=letter_choices, prefix=prompt_prefix,
            suffix=prompt_suffix
        )
        # Generate response from model using question + all choices
        response = ask(model, tokenizer, prompt)
        pred = perplexity_pred = best_choice_by_perplexity(
            model, tokenizer, question, answer_choices, letter_choices=letter_choices,
            question_choice_concat=question_choice_concat
        )
        # If model generates the text `answer_indicator` use the text generated after it
        # instead of lowest perplexity choice
        confident = False
        if answer_indicator in response:
            generated_answer = response[response.index(answer_indicator) + len(answer_indicator):]
            pred = generated_answer.lstrip(' ')[0].upper()
            confident = True

        answer_idx = answer_choice_dict["answer"]
        answer = chr(ORD_A + answer_idx) if letter_choices else answer_idx + 1
        answered_correctly = pred == answer

        if verbose:
            print(f"### Prompt ###\n{prompt}")
            print(f"### Generated ###\n{response.replace(prompt, '')}")
            print(f"### Predicted ###\n{pred}")
            print(f"### Answer ###\n{answer}")

        correct_count += answered_correctly  # Correct answer
        confident_correct_count += confident and answered_correctly  # Confident correct answers
        best_guess_correct_count += perplexity_pred == answer  # Educated guess
        num_questions += 1

    return [
        count / num_questions
        for count in [correct_count, confident_correct_count, best_guess_correct_count]
    ]

# Used to generate en_tw_translaion dataset (random seed = 1234)
def en_tw_question_gen(num_choices=4, num_questions=None):
    dataset_len = len(en_tw_dataset['ch'])
    generated = 0
    for i, en_text in enumerate(en_tw_dataset['en']):
        if num_questions and generated == num_questions:
            break
        correct_answer_text = en_tw_dataset['ch'][i]
        choices = [correct_answer_text]
        choices += [
            en_tw_dataset['ch'][r]
            for r in random.sample(range(dataset_len), num_choices - 1)
        ]
        random.shuffle(choices)
        answer_idx = choices.index(correct_answer_text)
        yield en_text, {"choices": choices, "answer": answer_idx}
        generated += 1

def create_pruning_mask(model):
  pruning_mask = {}
  for name, param in model.named_parameters():
      if 'weight' in name:  # Assuming you only pruned weights
          mask = param.data != 0  # True for unpruned weights
          pruning_mask[name] = mask
  return pruning_mask

def apply_pruning_mask(model, pruning_mask):
    for name, param in model.named_parameters():
        if name in pruning_mask and hasattr(param.grad, "data"): # For some reason some params have .grad.data = None when USE_8BIT = True
            param.grad.data.mul_(pruning_mask[name])

def train(model, dataloader, optimizer, scheduler, pruning_mask, dtype=None):
    model.train()
    if dtype is not None:
        model.to(dtype)
    total_loss = 0

    for batch in tqdm.tqdm(dataloader, desc='Training', leave=False):
        inputs = {k: v.to(model.device) for k, v in batch.items()}
        optimizer.zero_grad()

        # Forward pass
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
        total_loss += loss.item()

        # Backward and optimize
        loss.backward()
        apply_pruning_mask(model, pruning_mask)
        optimizer.step()
        scheduler.step()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate_ft(model, dataloader, dtype=None):
    model.eval()
    if dtype is not None:
        model.to(dtype)
    total_loss = 0
    n_samples = 0

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Evaluating"):
            inputs = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            total_loss += loss.item() * inputs['input_ids'].size(0)
            n_samples += inputs['input_ids'].size(0)
    avg_loss = total_loss / n_samples
    perplexity = torch.exp(torch.tensor(avg_loss))

    return perplexity.item()

class LLMTokenizedDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings.input_ids)

    def __getitem__(self, idx):
        # Directly return the sliced tensors, without creating new ones
        return {key: val[idx] for key, val in self.encodings.items()}
