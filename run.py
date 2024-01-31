import argparse
from evaluation import evaluate
from dataset import make_datasets

# Tested models
MODEL_PATHS = [
    "microsoft/phi-1_5",
    "facebook/opt-1.3b",
    "princeton-nlp/Sheared-LLaMA-1.3B"
]

parser = argparse.ArgumentParser(description='Run evaluation suite.')
parser.add_argument('model_path', type=str,
                    help=f'Huggingface model path to load. Tested with: ' + ', '.join(MODEL_PATHS))
parser.add_argument('--datasets', nargs='+', choices=['wiki_test', 'medical', 'legal', 'translation', 'economics', 'skyrim'],
                    default=['economics'],
                    help='Datasets to test')
parser.add_argument('--qdatasets', nargs='+', choices=['medical', 'translation', 'economics', 'skyrim'],
                    default=['economics'],
                    help='Question datasets to use for evaluation')
parser.add_argument('--use_8bit', type=bool,
                    default=False,
                    help='Use 8bit model and optimizer (Issues on Windows without Jupyter notebook)')
parser.add_argument('--lprune', type=float, nargs='+',
                    default=[1e-3],
                    help='Linear pruning thresholds to test')
parser.add_argument('--aprune', type=float, nargs='+',
                    default=[1e-3],
                    help='Activation pruning thresholds to test')
parser.add_argument('--eprune', type=float, nargs='+',
                    default=[-1],  # Don't prune embeddings
                    help='Embedding pruning thresholds to test')


if __name__ == "__main__":
    args = parser.parse_args()
    evaluate(
        [args.model_path],
        *make_datasets(
            datasets=args.datasets,
            question_datasets=args.qdatasets
        ),
        use8_bit=args.use_8bit,
        linear_options=args.lprune,
        activation_options=args.aprune,
        embedding_options=args.eprune
    )