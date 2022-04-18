import argparse
import json

from .dataset import dataset_dispatcher, separate_train_test
from .trainer import get_next_data_batch, plot_data_batch, score_data_batch
from .util import load_config


def compare_cifar(path, index, n_repeat, dataset_mode):
    model, args = load_config(path)
    args.dataset_mode = dataset_mode
    learn_loaders, _, _ = dataset_dispatcher(args)
    train_loader, test_loaders = separate_train_test(learn_loaders)
    train_loader, test_loaders = iter(train_loader), {k: iter(v) for k, v in test_loaders.items()}
    for _ in range(index+1):
        data_batch = get_next_data_batch(train_loader=train_loader, test_loaders=test_loaders)
    result, pred_batch = score_data_batch(data_batch=data_batch, model=model,
                                          acc_thresh=0.005, n_repeat=n_repeat)
    result = {k: round(v, 5) for k, v in result.items()}
    img = plot_data_batch(data_batch=pred_batch)
    print(json.dumps(result, sort_keys=True, indent=4))
    img.image.show()


if __name__ == "__main__":
    # Usage: python -m bayes_pcn.examine --path=...
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='runs/debug/latest.pt')
    parser.add_argument('--index', type=int, default=0, help='denotes which batch index to look.')
    parser.add_argument('--n-repeat', type=int, default=1)
    parser.add_argument('--dataset-mode', type=str, default='fast',
                        choices=['fast', 'mix', 'white', 'drop', 'mask'],
                        help='Specifies test dataset configuration.')
    args = parser.parse_args()
    compare_cifar(path=args.path, index=args.index, n_repeat=args.n_repeat,
                  dataset_mode=args.dataset_mode)
