import gc
import torch

from model_manager import ModelManager


def param_comb_generator(
        linear_options=[1e-5, 1e-3, 1e-1],
        activation_options=[1e-5, 1e-3, 1e-1],
        embedding_options=[-1, 0]
    ):
    for linear_thresh in linear_options:
        for activ_thresh in activation_options:
            for embed_thresh in embedding_options:
                yield linear_thresh, activ_thresh, embed_thresh


def evaluate(
        model_paths: [str],
        dataset_dict: dict,
        question_dataset_dict: dict,
        use8_bit=False,
        linear_options=[1e-5, 1e-3, 1e-1],
        activation_options=[1e-5, 1e-3, 1e-1],
        embedding_options=[-1, 0]
    ):
    eval_question_sets = list(question_dataset_dict.keys())
    for model_path in model_paths:        
        mm = ModelManager(model_path, dataset_dict, question_dataset_dict, use_8bit=use8_bit)
        for dataset_name in dataset_dict:
            print(f"Running eval {dataset_name} for {model_path}...")
            mm.calibrate(dataset_name, eval_question_sets=eval_question_sets)
            param_iterator = param_comb_generator(
                linear_options=linear_options,
                activation_options=activation_options,
                embedding_options=embedding_options
            )
            for linear_thresh, activ_thresh, embed_thresh in param_iterator:
                print(f"Eval for {dataset_name} with: linear_thresh={linear_thresh}, activ_thresh={activ_thresh}, embed_thresh={embed_thresh}")
                print("pruning...")
                mm.prune(
                    linear_thresh=linear_thresh, activ_thresh=activ_thresh,
                    embed_thresh=embed_thresh, prune_set_name=dataset_name,
                    run_eval=True, eval_question_sets=eval_question_sets
                )
                print("fine tuning...")
                mm.fine_tune(
                    dataset_name, batch_size=2048, perplexity_threshold="base_line",
                    max_epochs=200, pad_token=' ', verbose=False, run_eval=True,
                    eval_question_sets=eval_question_sets
                )
                mm.reset_model()
                mm.save_results()
        mm.reset()
        del mm
        gc.collect()
        torch.cuda.empty_cache()
