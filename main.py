import os
import warnings
import logging
import logging.handlers
import multiprocessing
import random
import json
import socket
from datetime import datetime
import time
from argparse import ArgumentParser
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from attacks.attacks import *
from models.statefuldefense import init_stateful_classifier
from utils import datasets

warnings.filterwarnings("ignore")


def main(args):
    # Set up logging and load config.
    if not args.disable_logging:
        random.seed(round(time.time() * 1000))
        log_dir = os.path.join("/".join(args.config.split("/")[:-1]), 'logs', args.config.split("/")[-1].split(".")[0])
        writer = SummaryWriter(log_dir=log_dir)
        logging.basicConfig(
            filename=os.path.join(writer.log_dir, f'log_{args.start_idx}_{args.start_idx + args.num_images}.txt'),
            level=logging.INFO)
        logging.info(args)

    config = json.load(open(args.config))
    model_config, attack_config = config["model_config"], config["attack_config"]

    logging.info(model_config)
    logging.info(attack_config)

    # Load model.
    model = init_stateful_classifier(model_config)
    model.eval()
    model.to("cuda")

    # Load dataset.
    if model_config["dataset"] == "mnist":
        transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
    elif model_config["dataset"] == "cifar10":
        transform = transforms.Compose([transforms.ToTensor()])
    elif model_config["dataset"] == "gtsrb":
        transform = transforms.Compose([transforms.ToTensor()])
    elif model_config["dataset"] == "imagenet":
        transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()])
    elif model_config["dataset"] == "iot_sqa":
        transform = transforms.Compose([transforms.ToTensor()])
    elif model_config["dataset"] == "celebahq":
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
    else:
        raise ValueError("Dataset not supported.")

    test_dataset = datasets.StatefulDefenseDataset(name=model_config["dataset"], transform=transform,
                                                   size=args.num_images, start_idx=args.start_idx)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    if attack_config["attack"] == "natural_accuracy":
        natural_performance(model, test_loader)
    else:
        attack_loader(model, test_loader, model_config, attack_config)


if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    parser = ArgumentParser()
    parser.add_argument('--disable_logging', action='store_true')
    parser.add_argument('--config', type=str)
    parser.add_argument('--num_images', type=int)
    parser.add_argument('--start_idx', type=int)
    parser.add_argument('--log_dir', type=str)
    main(parser.parse_args())
