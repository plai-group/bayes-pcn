import argparse
import json
import torch

from .dataset import dataset_dispatcher
from .trainer import get_next_data_batch, plot_data_batch, score_data_batch
from .util import load_config


def compare_cifar(path, index, n_repeat):
    model, args = load_config(path)
    train, tests, _ = dataset_dispatcher(args)
    data_batch = get_next_data_batch(iter(train), {k: iter(v) for k, v in tests.items()})
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
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--n-repeat', type=int, default=1)
    args = parser.parse_args()
    compare_cifar(path=args.path, index=args.index, n_repeat=args.n_repeat)
