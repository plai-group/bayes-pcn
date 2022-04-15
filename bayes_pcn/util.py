import io
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from PIL import Image
import torch
import torchvision
import wandb

from copy import deepcopy

from .dataset import dataset_dispatcher


def save_config(config, path):
    config = (deepcopy(config[0]), dict(config[1]))
    config[0].device = torch.device('cpu')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    with open(path, 'wb') as f:
        pickle.dump(config, f)


def load_config(path):
    with open(path, 'rb') as f:
        config = pickle.load(f)
        config = (config[0], DotDict(config[1]))
        return config


def unnormalize(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.cpu().numpy()
    return np.transpose(npimg, (1, 2, 0)).clip(0, 1)


def imshow(img):
    plt.imshow(unnormalize(img))
    plt.show()


def fig2img(fig, caption: str) -> wandb.Image:
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    pil_img = Image.open(buf)
    img = wandb.Image(pil_img, caption=caption)
    pil_img.close()
    return img


def save_result(result, path):
    if os.path.exists(path):
        prev_df = pd.read_csv(path)
        print("Will merge this previous result csv with the new result:")
        print(prev_df)
        df = pd.concat((prev_df, pd.DataFrame(result)))
    else:
        df = pd.DataFrame(result)
    print("Result csv:")
    print(df)
    df.to_csv(path, index=False)


def generate(path, d_batch):
    model, _ = load_config(path)
    X_top_down, _ = model.generate_ancestral(d_batch=d_batch, noise=0.05)
    X_gen, _ = model.generate_iterative(d_batch=d_batch, noise=0.05)
    X_total = torch.cat((X_top_down, X_gen), dim=0)
    imshow(torchvision.utils.make_grid(X_total.reshape(-1, 3, 32, 32), nrow=d_batch))


class DotDict(dict):
    """
    https://stackoverflow.com/questions/13520421/recursive-dotdict/13520518
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        if dct:
            for key, value in dct.items():
                if hasattr(value, 'keys'):
                    value = DotDict(value)
                elif isinstance(value, list):
                    value = [x for x in value]
                self[key] = value

    def get_dict(self):
        ret = {}
        for key, value in self.items():
            if hasattr(value, 'keys'):
                value = value.getDict()
            ret[key] = value
        return ret
