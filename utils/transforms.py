import torch
import numpy as np
import torchvision


def random_affine(x, angle, scale, tx, ty):
    x = torchvision.transforms.RandomAffine(degrees=angle, translate=(tx, ty), scale=(scale, scale))(x)
    return x


def transform(x, **kwargs):
    if kwargs['type'] == 'random_affine':
        return random_affine(x, kwargs['angle'], kwargs['scale'], kwargs['tx'], kwargs['ty'])
    else:
        raise NotImplementedError
