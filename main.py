"""
Author: Caleb Ehrisman
Course- CSC-514 Computer Vision
Assignment - Semester Project - Text Transcription from images
"""

import argparse
import os
import sys

import torch.utils.data
import numpy as np
from process import Processor
from bounds import bound_region, get_letters, get_lines
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
    # model = torch.load("model/model.pth")

    get_lines(args.image)
    bound_region("lines")
    get_letters("words", )



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

    # dataset = cnn.CustomImageDataSet()
    # batch_size = 4
    # validation_split = .2
    # shuffle_dataset = True
    # random_seed = 42
    #
    # # Creating data indices for training and validation splits:
    # dataset_size = len(dataset)
    # indices = list(range(dataset_size))
    # split = int(np.floor(validation_split * dataset_size))
    # if shuffle_dataset:
    #     np.random.seed(random_seed)
    #     np.random.shuffle(indices)
    # train_indices, val_indices = indices[split:], indices[:split]
    #
    # # Creating PT data samplers and loaders:
    # train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    # valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
    #
    # train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
    #                                            sampler=train_sampler)
    # validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
    #                                                 sampler=valid_sampler)
    #
    # # data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    #
    # cnn_model = cnn.CNN()
    # cnn.train(30, cnn_model, train_loader)
    #
    # cnn.test(cnn_model, validation_loader)
    # torch.save(cnn_model.state_dict(), "model/model.pth")
    saved = torch.load("model/model.pth")

    cnn_model_test = cnn.CNN()
    cnn_model_test.load_state_dict(saved)

    test = cnn.ImageDataSetToClassify()
    test1 = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False)
    cnn.classify(cnn_model_test, test1)


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
