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
import glob
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

    # show_img(erosion)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (4, 5))
    dilate = cv2.dilate(erosion, kernel, iterations=5)

    # show_img(dilate)

    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if os.path.exists('samples'):
        shutil.rmtree('samples')
    os.mkdir(cwd + '/samples')

    contours, boxes = sort_words(contours)

    i = 0
    for cnt in contours:
        [x, y, w, h] = cv2.boundingRect(cnt)
        roi = img_blown[y:y + h, x:x + w]
        cv2.imwrite('samples/Boxed_ROIs' + str(i) + '.jpg', roi)
        output = cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 1)
        i += 1

    # img = cv2.resize(output, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
    # show_img(img)


def get_letters(image_path):
    cwd = os.getcwd()
    if os.path.exists('letters'):
        shutil.rmtree('letters')
    os.mkdir(cwd + '/letters')

    i = 0
    file_num = 0
    img = cv2.imread(image_path + '\Boxed_ROIs{}.jpg'.format(file_num))
    while img is not None:

        scale_percent = 200
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img_blown = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        output = img_blown.copy()

        img_gray = cv2.cvtColor(img_blown, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((2, 1), np.uint8)
        ret, thresh = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY_INV)
        erosion = cv2.erode(thresh, kernel, iterations=3)

        # cv2.imshow("test", erosion)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 3))
        dilate = cv2.dilate(erosion, kernel, iterations=8)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 1))
        dilate = cv2.dilate(dilate, kernel, iterations=2)
        # cv2.imshow("test", dilate)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        contours, boxes = sort_letters(contours)

        for cnt in contours:
            [x, y, w, h] = cv2.boundingRect(cnt)
            roi = img_blown[y:y + h, x:x + w]
            cv2.imwrite('letters/Boxed_ROIs_letters' + str(i) + '.jpg', roi)
            output = cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 1)
            i += 1

        file_num += 1
        img = cv2.imread(image_path + '\Boxed_ROIs{}.jpg'.format(file_num))
        print(image_path + '\Boxed_ROIs{}.jpg'.format(file_num))
        # img = cv2.resize(output, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
        # show_img(img)


def show_img(img):
    cv2.imshow("test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def sort_words(contours):
    boxes = [cv2.boundingRect(c) for c in contours]

    sortedx = zip(*sorted(zip(contours, boxes), key=lambda b: b[1][0], reverse=False))
    (contours, boxes) = zip(*sorted(zip(*sortedx), key=lambda b: b[1][1], reverse=False))

    return (contours, boxes)


def sort_letters(contours):
    boxes = [cv2.boundingRect(c) for c in contours]
    (contours, boxes) = zip(*sorted(zip(contours, boxes), key=lambda b: b[1][0], reverse=False))

    return (contours, boxes)
