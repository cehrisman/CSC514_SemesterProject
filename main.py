"""
Author: Caleb Ehrisman
Course- CSC-514 Computer Vision
Assignment - Semester Project - Text Transcription from images
"""

import argparse
import os
import sys

import torch.utils.data

from process import Processor
from bounds import bound_region, get_letters
from matplotlib import pyplot as plt
import cv2
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import cnn

"""
    parse gathers command line arguments.

    :return: a list of all parsed arguments
"""


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Path for image file to load')
    return parser.parse_args()


def main():
    args = check_args()
    # process = Processor(args.image)
    # img = process.load_image()
    # img = process.sharpen(img)
    # img = process.threshhold(img)
    bound_region(args.image)
    get_letters("samples")
    # train_data = datasets.MNIST(
    #     root='data',
    #     train=True,
    #     transform=transforms.ToTensor(),
    #     download=True,
    # )
    #
    # test_data = datasets.MNIST(
    #     root='data',
    #     train=False,
    #     transform=transforms.ToTensor()
    # )
    #
    # loaders = {
    #     'train': torch.utils.data.DataLoader(train_data,
    #                                          batch_size=100,
    #                                          shuffle=True,
    #                                          num_workers=1),
    #     'test': torch.utils.data.DataLoader(train_data,
    #                                         batch_size=100,
    #                                         shuffle=True,
    #                                         num_workers=1),
    # }

    # cnn_model = cnn.CNN()
    # cnn.train(10, cnn_model, loaders)
    #
    # cnn.test(cnn_model, loaders)


def check_args():
    args = parse()
    path = None

    if args.image is not None:
        path = args.image

    if path is None:
        sys.exit("No file path entered.")

    if not os.path.exists(path):
        sys.exit(path + " File not found. Check file path name")

    return args


if __name__ == "__main__":
    main()
    # Error checking for path input and checking if file exists.
    # If here then file exists
