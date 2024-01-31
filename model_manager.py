import os
import gc
import copy
import json
import torch

import pandas as pd
_8BIT_AVAILABLE = False
try:
    import bitsandbytes as bnb
    _8BIT_AVAILABLE = True
except:
    print("Failed to load bitsandbytes, 8bit not available.")
    print("transformers library may fail to load in Windows when bitsandbytes is installed and code is not being run from Jupyter Notebook")
import transformers as tfmr

from datetime import datetime
from torch.utils.data import DataLoader
from prune import (
    prune_linear, prune_activations, prune_embedding
)
from utils import (
    LLMTokenizedDataset, ask, get_special_tokens, get_calib_feat, flatten_neuron_dict,
    count_all_tokens, collect_stats, evaluate, create_pruning_mask, model_question_eval,
    train, evaluate_ft, save_json
)


class ModelManager:
    def __init__(self, model_path, dataset_dict, question_dataset_dict, use_8bit=True, **model_kwargs):
        self.tokenizer = tfmr.AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.special_tokens = get_special_tokens(self.tokenizer)
        self.use_8bit = use_8bit and _8BIT_AVAILABLE
        self.model_path = model_path
        self.model = self.load_base_model(model_path=None, **model_kwargs)
        self.results_dict = {}
        self.all_neuron_df = None
        self._pad_token_backup = None
        self.prune_state = {}
        self.calib_set = None
        self.dataset_label = "datasets"
        self.question_label = "multiple_choice"
        self.fine_tune_label = "fine_tune"
        self.fine_tune_stats_label = "fine_tune_stats"
        self.DATASETS = dataset_dict
        self.QUESTION_DATASETS = question_dataset_dict
    
    def __call__(self, prompt, max_new=10):
        return ask(self.model, self.tokenizer, prompt, max_new=max_new)
    
    @property
    def is_calibrated(self):
        return (self.all_neuron_df is not None) and (self.calib_set is not None)
    
    def load_base_model(self, path=None, **model_kwargs):
        model_path = self.model_path if path is None else path
        return tfmr.AutoModelForCausalLM.from_pretrained(
            model_path, device_map="cuda:0", trust_remote_code=model_kwargs.get("trust_remote_code", True),
            load_in_8bit=self.use_8bit
        ).eval()
    
    def reset_model(self, use_8bit=None):
        self.prune_state = {}
        self._pad_token_backup = None
        if use_8bit is not None:
            self.use_8bit = use_8bit
        try:
            self.model.to("cpu")
        except:
            pass
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        self.model = self.load_base_model()
    
    def reset(self):
        self.all_neuron_df = None
        self.calib_set = None
        self.reset_model()
    
    def convert_dataset(self, dataset_name):
        dataset = self.DATASETS[dataset_name].shuffle(seed=42)
        test_data = self.tokenizer("\n\n".join(dataset['text']), return_tensors='pt').input_ids.to(self.model.device)
        return test_data
    
    def calibrate(self, dataset_name, reset_model=False, verbose=False, run_eval=True, eval_question_sets=[]):
        if reset_model:
            self.reset_model()
        all_dict = dict()

        test_data = self.convert_dataset(dataset_name)
        neuron_dicts = get_calib_feat(self.model, test_data, all_dict, device=self.model.device)

        # Flatten the neuron dictionary
        neuron_type = 'activations'
        activation_df = flatten_neuron_dict(neuron_dicts[neuron_type], neuron_type=neuron_type)
        neuron_type = 'linear_input'
        linear_input_df = flatten_neuron_dict(neuron_dicts[neuron_type], neuron_type=neuron_type)
        neuron_type = 'linear_output'
        linear_output_df = flatten_neuron_dict(neuron_dicts[neuron_type], neuron_type=neuron_type)
        all_neuron_df = pd.concat([linear_input_df, linear_output_df, activation_df], axis=0)
        all_neuron_df = all_neuron_df.assign(SmallestAverageMagnitude = all_neuron_df[
            ["AverageMagnitude", "UnbiasedAverageMagnitude"]
        ].min(axis=1))
        if verbose:
            print(all_neuron_df.shape)
        assert all_neuron_df['Count'].nunique() == 1
        all_neuron_df.head(3)
        self.all_neuron_df = all_neuron_df
        self.calib_set = dataset_name
        if run_eval:
            self.eval(dataset_name, rerun=True)
            for question_set in eval_question_sets:
                self.question_eval(question_set, rerun=True)
            gc.collect()
            torch.cuda.empty_cache()
    
    def prune(self, linear_thresh=0, activ_thresh=0, embed_thresh=-1, prune_set_name=None, run_eval=True, eval_question_sets=[]):
        dataset_name = self.calib_set if prune_set_name is None else prune_set_name
        orig_prune_state = copy.deepcopy(self.prune_state)
        if linear_thresh:
            self.prune_state["linear_thresh"] = linear_thresh
            self.model = prune_linear(
                self.model, self.all_neuron_df.query(f"NeuronType in {['linear_input','linear_output']}"),
                threshold=linear_thresh
            )
        if activ_thresh:
            self.prune_state["activ_thresh"] = activ_thresh
            self.model = prune_activations(
                self.model, self.all_neuron_df.query(f"NeuronType in {['activations']}"),
                threshold=activ_thresh, mapping=['act', 'fc1']
            )
        if embed_thresh >= 0:
            self.prune_state["embed_thresh"] = embed_thresh
            self.prune_state["embed_prune_set"] = dataset_name
            calib_set = self.convert_dataset(dataset_name)
            token_counts = count_all_tokens(calib_set, self.tokenizer)
            normalized_counts = token_counts / token_counts.sum()
            self.model = prune_embedding(
                self.model, normalized_counts, self.special_tokens,
                threshold=embed_thresh, names = ['transformer.embd.wte', 'lm_head.linear']
            )
        if self.prune_state != orig_prune_state and run_eval:
            self.eval(dataset_name)
            for question_set in eval_question_sets:
                self.question_eval(question_set)
            gc.collect()
            torch.cuda.empty_cache()
    
    def get_state_name(self, linear_thresh=None, activ_thresh=None, embed_thresh=None, embed_prune_set=None, calib_set=None, precision=None):
        linear_thresh = linear_thresh if linear_thresh is not None else self.prune_state.get('linear_thresh', 0)
        activ_thresh = activ_thresh if activ_thresh is not None else self.prune_state.get('activ_thresh', 0)
        embed_thresh = embed_thresh if embed_thresh is not None else self.prune_state.get('embed_thresh', -1)
        embed_prune_set = embed_prune_set if embed_prune_set is not None else self.prune_state.get('embed_prune_set', '')
        calib_set = calib_set if calib_set is not None else self.calib_set
        precision = precision if precision is not None else (8 if self.use_8bit else 32)
        state_name = f"linear={linear_thresh},activ={activ_thresh},embed={embed_thresh},embed_prune_set={embed_prune_set},calib={calib_set},prec={precision}"
        return state_name
    
    def current_state_name(self):
        return self.get_state_name()
    
    def base_state_name(self):
        return self.get_state_name(linear_thresh=0, activ_thresh=0, embed_thresh=-1, embed_prune_set='')
    
    def get_state(self, **kwargs):
        state_name = self.get_state_name(**kwargs)
        self.results_dict.setdefault(state_name, {})
        return self.results_dict[state_name]
    
    def get_base_state(self):
        return self.results_dict[self.base_state_name()]
    
    def question_eval(self, dataset_name, rerun=False):
        state_dict = self.get_state()
        fine_tuned = self.fine_tune_label in state_dict
        if fine_tuned:
            state_dict.setdefault(self.fine_tune_label, {})
            state_dict = state_dict[self.fine_tune_label]
        elif "size_stats" not in state_dict:
            state_dict["size_stats"] = collect_stats(self.model)
        state_dict.setdefault(self.question_label, {})
        question_results = state_dict[self.question_label]

        if dataset_name in question_results and not rerun:
            return question_results[dataset_name] if not fine_tuned else question_results[dataset_name][-1]
        
        correct, correct_generated, correct_best_choice = model_question_eval(
            self.model, self.tokenizer, self.QUESTION_DATASETS[dataset_name]
        )
        result = {
            "correct": correct,
            "correct_generated": correct_generated,
            "correct_best_choice": correct_best_choice
        }
        if not fine_tuned:
            question_results[dataset_name] = result
        else:
            question_results.setdefault(dataset_name, [])
            question_results[dataset_name].append(result)
        return result
    
    def eval(self, dataset_name, batch_size=2048, eval_func=evaluate, rerun=False):        
        # Setup dataset
        test_data = self.convert_dataset(dataset_name)

        state_dict = self.get_state()
        fine_tuned = self.fine_tune_label in state_dict
        if fine_tuned:
            state_dict.setdefault(self.fine_tune_label, {})
            state_dict = state_dict[self.fine_tune_label]
        elif "size_stats" not in state_dict:
            state_dict["size_stats"] = collect_stats(self.model)
        
        state_dict.setdefault(self.dataset_label, {})
        dataset_results = state_dict[self.dataset_label]

        # Evaluate on dataset
        result = dataset_results.get(dataset_name)
        if result is None or rerun:
            result = eval_func(self.model, test_data, batch_size=batch_size, device=self.model.device).item()
            if not fine_tuned:
                dataset_results[dataset_name] = result
            else:
                dataset_results.setdefault(dataset_name, [])
                dataset_results[dataset_name].append(result)
            
        return result
    
    def save_results(self, save_path="auto", indent_size=4):
        if save_path is None or save_path == "auto":
            save_dir = "model_man"
            os.makedirs(save_dir, exist_ok=save_path == "auto")
            curr_time = datetime.now()
            save_path = os.path.join(save_dir, curr_time.strftime("%m-%d-%Y_%H-%M-%S") + ".json")
        save_json(self.results_dict, save_path, indent_size=indent_size)
    
    def load_results(self, results_path):
        with open(results_path, 'r', encoding="utf-8") as file:
            self.results_dict = json.load(file)
    
    def fine_tune(
            self, dataset_name, batch_size=2048, perplexity_threshold="base_line",
            train_func=train, eval_func=evaluate_ft, max_epochs=200, pad_token=' ',
            verbose=True, run_eval=True, eval_question_sets=[]):
        current_state_dict = self.get_state()
        current_state_dict.setdefault(self.fine_tune_label, {})
        fine_tune_dict = current_state_dict[self.fine_tune_label]
        fine_tune_dict.setdefault("tuned_on", [])
        
        dataset_ft = self.DATASETS[dataset_name].shuffle(seed=42)

        # Setup tokenizer
        self._pad_token_backup = self.tokenizer.pad_token
        if pad_token is not None:
            self.tokenizer.pad_token = pad_token
        tokenized_inputs = self.tokenizer(
            "\n\n".join(dataset_ft['text']), return_tensors="pt", truncation=True,
            max_length=512, padding="max_length"
        )
        tokenized_dataset = LLMTokenizedDataset(tokenized_inputs)
        dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)
        
        optimizer_type = bnb.optim.AdamW8bit if self.use_8bit else torch.optim.AdamW
        dtype = None if self.use_8bit else torch.bfloat16  # .to(dtype) does not work for load_in_8bit models
        
        optimizer = optimizer_type(self.model.parameters(), lr=5e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        pruning_mask = create_pruning_mask(self.model)
        threshold_val = perplexity_threshold
        if perplexity_threshold == "base_line":
            threshold_val = self.get_base_state()[self.dataset_label][dataset_name]
        
        epochs = 0
        while epochs < max_epochs:
            perplexity = eval_func(self.model, dataloader, dtype=dtype)
            if perplexity <= threshold_val:
                break
            torch.cuda.empty_cache()
            avg_loss = train_func(self.model, dataloader, optimizer, scheduler, pruning_mask, dtype=dtype)            
            if verbose:
                print(f'Epoch {epochs + 1}: Avg Loss {avg_loss:.2f}, Perplexity {perplexity:.2f}')
            epochs += 1
        
        fine_tune_dict["tuned_on"].append({
            "dataset": dataset_name,
            "epochs": epochs,
            "max_epocs": max_epochs,
            "perplexity": perplexity,
            "perplexity_thresh": threshold_val,
        })
        gc.collect()
        torch.cuda.empty_cache()
        self.tokenizer.pad_token = self._pad_token_backup
        if not self.use_8bit:
            self.model.to(torch.float32)
        rerun = True
        if run_eval:
            self.eval(dataset_name, rerun=rerun)
            for question_set in eval_question_sets:
                self.question_eval(question_set, rerun=rerun)
            gc.collect()
            torch.cuda.empty_cache()
