from PIL import Image
import torch
import torch.distributions as dists
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from typing import Dict, Tuple, Any, Optional, Callable


class PureNoise:
    def __call__(self, tensor):
        # result = torch.randn(tensor.size()) * (self.var) ** 0.5 + self.mean
        result = torch.rand(tensor.shape)
        return result, torch.zeros(tensor.shape)


class WhiteNoise:
    def __init__(self, mean=0., var=0.2):
        self.var = var
        self.mean = mean

    def __call__(self, tensor):
        result = tensor + torch.randn(tensor.size()) * (self.var) ** 0.5 + self.mean
        return result, torch.zeros(tensor.shape)


class DropoutNoise:
    def __init__(self, p, n_channels=3):
        self.p = p
        self.n_channels = n_channels

    def __call__(self, tensor):
        if self.n_channels == 0:
            mask = dists.Bernoulli(torch.ones(tensor.shape) * (1-self.p)).sample()
        else:
            mask = dists.Bernoulli(torch.ones(tensor.size()[1:]) * (1-self.p)).sample()
            mask = mask.unsqueeze(0).repeat((self.n_channels, 1, 1))
        return tensor * mask, mask


class MaskingNoise:
    def __init__(self, p, n_channels=3):
        self.p = p
        self.n_channels = n_channels

    def __call__(self, tensor):
        if self.n_channels == 0:
            mask = torch.ones(tensor.shape).flatten()
            mask[:int(len(mask) * self.p)] = 0.
            mask = mask.reshape(tensor.shape)
        else:
            width, height = tensor.shape[1:]
            n_masked_cols = int(width * self.p)
            n_visible_cols = width - n_masked_cols
            mask = torch.cat((torch.ones(height, n_visible_cols),
                             torch.zeros(height, n_masked_cols)),
                             dim=1).repeat((self.n_channels, 1, 1))
        return tensor * mask, mask


class CIFAR10Recall(datasets.CIFAR10):
    """Wrapper around CIFAR10 dataset that, instead of returning target image class, returns
    image indices to hold fixed during recall if applicable. These indices can be pixels that
    we know to be unnoised.
    """
    def __init__(self, root: str, train: bool = True, noise_transform: Optional[Callable] = None,
                 transform: Optional[Callable] = None, transform_post: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, download: bool = False) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.noise_transform = noise_transform
        self.transform_post = transform_post

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = self.data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.noise_transform is not None:
            img, fixed_indices = self.noise_transform(img)
        else:
            fixed_indices = torch.zeros(img.shape)

        if self.transform_post is not None:
            img = self.transform_post(img)

        return img, fixed_indices


def cifar10(args) -> Tuple[Dict[str, DataLoader], Dict[str, DataLoader], Dict[str, Any]]:
    data_size = args.n_data
    learn_batch_size = min(args.n_batch, data_size)
    score_batch_size = min(args.n_batch_score, data_size)
    assert (data_size % learn_batch_size) == 0 and (data_size % score_batch_size) == 0
    x_dim = 3 * 32 * 32

    config = args.dataset_mode

    # Image preprocessing logic
    transform = transforms.Compose([transforms.ToTensor()])
    transform_post = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if config == 'fast':
        noise_transforms = {'mask0.25': MaskingNoise(p=0.25)}
    elif config == 'mix':
        noise_transforms = {'white0.2': WhiteNoise(var=0.2**2),
                            'drop0.25': DropoutNoise(p=0.25),
                            'mask0.25': MaskingNoise(p=0.25)}
    elif config == 'white':
        noise_transforms = {'white0.2': WhiteNoise(var=0.2**2),
                            'white0.4': WhiteNoise(var=0.4**2),
                            'white0.8': WhiteNoise(var=0.8**2)}
    elif config == 'drop':
        noise_transforms = {'drop0.25': DropoutNoise(p=0.25),
                            'drop0.50': DropoutNoise(p=0.50),
                            'drop0.75': DropoutNoise(p=0.75)}
    elif config == 'mask':
        noise_transforms = {'mask0.25': MaskingNoise(p=0.25),
                            'mask0.50': MaskingNoise(p=0.50),
                            'mask0.75': MaskingNoise(p=0.75)}
    else:
        raise Exception(f"dataset-mode '{config}' is not supported.")
    # if not config == 'fast':
    #     noise_transforms['pure'] = PureNoise()

    # Initialize train and test dataloaders
    train = CIFAR10Recall(root='./data', train=True, download=True,
                          transform=transform, transform_post=transform_post)
    train = torch.utils.data.Subset(train, range(data_size))
    tests = []
    learn_loaders = dict(train=DataLoader(train, batch_size=learn_batch_size, shuffle=False))
    score_loaders = dict(train=DataLoader(train, batch_size=score_batch_size, shuffle=False))
    for name, noise_transform in noise_transforms.items():
        test = CIFAR10Recall(root='./data', train=True, noise_transform=noise_transform,
                             transform=transform, transform_post=transform_post, download=False)
        test = torch.utils.data.Subset(test, range(data_size))
        tests.append(test)
        learn_loaders[f"test_{name}"] = DataLoader(test, batch_size=learn_batch_size, shuffle=False)
        score_loaders[f"test_{name}"] = DataLoader(test, batch_size=score_batch_size, shuffle=False)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    info = {'classes': classes, 'train': train, 'test': tests, 'x_dim': x_dim}
    return learn_loaders, score_loaders, info


def separate_train_test(loaders: Dict[str, DataLoader]) -> Tuple[DataLoader, Dict[str, DataLoader]]:
    train_loader = loaders['train']
    test_loaders = {name: loader for name, loader in loaders.items() if name != 'train'}
    return train_loader, test_loaders


def dataset_dispatcher(args):
    if args.dataset == 'cifar10':
        return cifar10(args)
    else:
        raise NotImplementedError()
