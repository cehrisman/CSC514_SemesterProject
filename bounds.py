"""
Author: Caleb Ehrisman
Course- CSC-514 Computer Vision
Assignment - Semester Project - Text Transcription from images

This file contains the code to identify regions of interest for text.
"""
import cv2
from PIL import Image
import numpy as np
import os
import shutil
from tensorflow.keras.preprocessing import image
from matplotlib import pyplot as plt


def bound_region(image_path):
    cwd = os.getcwd()

    img = cv2.imread(image_path)
    scale_percent = 300
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img_blown = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    output = img_blown.copy()


    img_gray = cv2.cvtColor(img_blown, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((2, 1), np.uint8)
    ret, thresh = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY_INV)
    erosion = cv2.erode(thresh, kernel, iterations=4)

    show_img(erosion)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (4, 5))
    dilate = cv2.dilate(erosion, kernel, iterations=5)
    cv2.imshow("test", dilate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    rect_d = []
    if os.path.exists('samples'):
        shutil.rmtree('samples')
    os.mkdir(cwd + '/samples')
    i = 0
    for cnt in contours:
        [x, y, w, h] = cv2.boundingRect(cnt)
        #if w < 20 and h < 20:
        #    continue
        roi = img_blown[y:y + h, x:x + w]
        cv2.imwrite('samples/Boxed_ROIs' + str(i) + '.jpg', roi)
        output = cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 1)
        i += 1
    img = cv2.resize(output, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
    cv2.imshow("test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_letters(image_path):
    cwd = os.getcwd()

    img = cv2.imread(image_path)
    scale_percent = 200
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    output = img.copy()

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((2, 1), np.uint8)
    ret, thresh = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY_INV)
    erosion = cv2.erode(thresh, kernel, iterations=4)

    cv2.imshow("test", erosion)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    cv2.imshow("test", dilate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    rect_d = []
    if os.path.exists('samples'):
        shutil.rmtree('samples')
    os.mkdir(cwd + '/samples')
    i = 0
    for cnt in contours:
        [x, y, w, h] = cv2.boundingRect(cnt)
        # if w < 20 and h < 20:
        #    continue
        roi = img[y:y + h, x:x + w]
        cv2.imwrite('samples/Boxed_ROIs' + str(i) + '.jpg', roi)
        output = cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 1)
        i += 1

    cv2.imshow("test", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_img(img):
    cv2.imshow("test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
