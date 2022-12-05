"""
Author: Caleb Ehrisman
Course- CSC-514 Computer Vision
Assignment - Semester Project - Text Transcription from images

This file handles the preprocessing needed
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image


class Processor:
    def __init__(self, file):
        self.file = file

    def load_image(self):
        img = cv2.imread(self.file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def sharpen(self, img):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        img = cv2.filter2D(img, -1, kernel)
        plt.imshow(img)
        plt.show()
        return img

    def threshhold(self, img):
        img = image.img_to_array(img, dtype='uint8')
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        img = 255 - img
        plt.imshow(img)
        plt.show()
