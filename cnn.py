"""
Author: Caleb Ehrisman
Course- CSC-514 Computer Vision
Assignment - Semester Project - Text Transcription from images

This file hosts the CNN class to classify text in images
"""

import torch
import torchvision
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torch import nn, optim
import glob


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = self.conv_module(1, 16)
        self.layer2 = self.conv_module(16, 32)
        self.layer3 = self.conv_module(32, 64)
        self.layer4 = self.conv_module(64, 128)
        self.layer5 = self.conv_module(128, 256)
        self.gap = self.global_avg_pool(256, 26)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.gap(out)
        out = out.view(-1, 26)

        return out

    def conv_module(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

    def global_avg_pool(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1)))


def train(num_epochs, cnn, loaders):
    optimizer = optim.Adam(cnn.parameters(), lr=0.001)
    cnn.train()

    total_step = len(loaders)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders):

            output = cnn(images)
            # print(torch.max(b_y, 0)[1])
            loss = loss_func(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}] , Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step,
                                                                          loss.item()))

    return cnn


def test(cnn, loaders):
    cnn.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders:
            test_output = cnn(images)
            _, pred_y = torch.max(test_output.data, 1)
            total += len(labels)
            print(pred_y)
            correct += (pred_y == labels).sum().item()

    print('Test Accuracy of the model on the {} test images: {} '.format(total, 100 * correct / total))


def classify(cnn, loader):
    cnn.eval()
    alpha_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l',
                  12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w',
                  23: 'x', 24: 'y', 25: 'z'}
    word = None
    with torch.no_grad():
        with open('output.txt', 'w') as f:
            for images, labels in loader:
                if word is None:
                    word = labels[0][0]
                if word is not labels[0][0]:
                    f.write(' ')
                    word = labels[0][0]
                output = cnn(images)
                _, predictions = torch.max(output.data, 1)
                text = [predictions.item()]
                letter = [alpha_dict[predictions.item()] for x in text]
                f.write(letter[0])


class CustomImageDataSet(Dataset):
    def __init__(self):
        self.imgs_path = "data/"
        file_list = glob.glob(self.imgs_path + "*")
        print(file_list)
        self.data = []
        for class_path in file_list:
            class_name = class_path.split('\\')[1]
            for img_path in glob.glob(class_path + "/*.jpg"):
                self.data.append([img_path, class_name])

        self.class_map = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7,
                          "i": 8, "j": 9, "k": 10, "l": 11, "m": 12, "n": 13, "o": 14, "p": 15,
                          "q": 16, "r": 17, "s": 18, "t": 19, "u": 20, "v": 21, "w": 22, "x": 23, "y": 24, "z": 25}
        self.img_dim = (128, 128)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        image = Image.open(img_path)
        image = ImageOps.grayscale(image)
        image = image.resize(self.img_dim)
        transform = torchvision.transforms.Compose([torchvision.transforms.PILToTensor()])
        img_tensor = transform(image)
        label = self.class_map[class_name]
        # img_tensor = img_tensor.permute(2, 0, 1)

        return img_tensor.float(), label


class ImageDataSetToClassify(Dataset):
    def __init__(self):
        self.imgs_path = "squared/"
        file_list = glob.glob(self.imgs_path + "*")
        print(file_list)
        self.data = []
        for class_path in file_list:
            class_name = class_path.split('\\')[1]
            print(class_name)
            for img_path in glob.glob(class_path + "/*.jpg"):
                self.data.append([img_path, class_name])

        self.img_dim = (128, 128)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        image = Image.open(img_path)
        image = ImageOps.grayscale(image)
        image = image.resize(self.img_dim)
        transform = torchvision.transforms.Compose([torchvision.transforms.PILToTensor()])
        img_tensor = transform(image)
        label = [class_name]
        return img_tensor.float(), label
        # img_tensor = img_tensor.permute(2, 0, 1)
