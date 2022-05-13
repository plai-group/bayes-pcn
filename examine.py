import argparse
import json

from bayes_pcn.dataset import dataset_dispatcher, separate_train_test
from bayes_pcn.trainer import get_next_data_batch, plot_data_batch,\
                              score_data_batch, generate_samples
from bayes_pcn.util import load_config


"""
TLDR: Used for debugging or quick visualization.

Load a trained model and print its performance on a specific train/test batch.
Test dataloaders don't have to be the same as one used while training.

Usage: `python examine.py --path=...`
"""


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--index', type=int, default=0,
                        help='denotes which batch index to look at.')
    parser.add_argument('--n-repeat', type=int, default=1)
    parser.add_argument('--dataset-mode', type=str, default='fast',
                        choices=['fast', 'mix', 'white', 'drop', 'mask', 'all'],
                        help='Specifies test dataset configuration.')
    return parser


def compare_cifar(path, index, n_repeat, dataset_mode):
    model, args = load_config(path)
    args.dataset_mode = dataset_mode
    learn_loaders, _, _ = dataset_dispatcher(args)
    train_loader, test_loaders = separate_train_test(learn_loaders)
    train_loader, test_loaders = iter(train_loader), {k: iter(v) for k, v in test_loaders.items()}
    for _ in range(index+1):
        data_batch = get_next_data_batch(train_loader=train_loader, test_loaders=test_loaders)
    # model.learn(data_batch.train[0])
    result, pred_batch = score_data_batch(data_batch=data_batch, model=model,
                                          acc_thresh=0.005, n_repeat=n_repeat)
    result = {k: round(v, 5) for k, v in result.items()}
    batch_img = plot_data_batch(data_batch=pred_batch)
    print(json.dumps(result, sort_keys=True, indent=4))
    batch_img.image.show()
    gen_img = generate_samples(model=model, X_shape=data_batch.original_shape, d_batch=8)
    gen_img.image.show()


if __name__ == "__main__":
    args = get_parser().parse_args()
    compare_cifar(path=args.model_path, index=args.index, n_repeat=args.n_repeat,
                  dataset_mode=args.dataset_mode)
