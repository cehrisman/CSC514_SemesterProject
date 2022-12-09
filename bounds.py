"""
Author: Caleb Ehrisman
Course- CSC-514 Computer Vision
Assignment - Semester Project - Text Transcription from images

This file contains the code to identify regions of interest for text.
"""
import cv2
from PIL import Image
import os
import shutil



def bound_region(image_path):
    cwd = os.getcwd()
    if os.path.exists('words'):
        shutil.rmtree('words')
    os.mkdir(cwd + '/words')

    i = 0
    file_num = 0
    img = cv2.imread(image_path + '\Boxed_ROIs{}.jpg'.format(file_num))
    while img is not None:
        img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (3, 5), 0)
        ret, thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.imwrite('words/Boxed_ROIs_dilate{}.jpg'.format(i), thresh)
        contours, boxes = sort_x(contours)

        for cnt in contours:
            [x, y, w, h] = cv2.boundingRect(cnt)
            if w > 10 or h > 10:
                roi = img[y:y + h, x:x + w]

                cv2.imwrite('words/Boxed_ROIs' + str(i) + '.jpg', roi)
                i += 1

        file_num += 1
        img = cv2.imread(image_path + '\Boxed_ROIs{}.jpg'.format(file_num))

def get_lines(image_path):
    cwd = os.getcwd()
    if os.path.exists('lines'):
        shutil.rmtree('lines')
    os.mkdir(cwd + '/lines')

    img = cv2.imread(image_path)
    output = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    thresh = cv2.dilate(thresh, kernel, iterations=3)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours, boxes = sort_y(contours)
    i = 0
    for cnt in contours:
        [x, y, w, h] = cv2.boundingRect(cnt)
        if w > 20 or h > 20:
            roi = img[y:y + h, x:x + w]
            cv2.imwrite('lines/Boxed_ROIs' + str(i) + '.jpg', roi)
            output = cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 1)
            i += 1


def get_letters(image_path):
    cwd = os.getcwd()
    if os.path.exists('letters'):
        shutil.rmtree('letters')
    os.mkdir(cwd + '/letters')
    if os.path.exists('squared'):
        shutil.rmtree('squared')
    os.mkdir(cwd + '/squared')

    i = 0
    file_num = 0
    img = cv2.imread(image_path + '\Boxed_ROIs{}.jpg'.format(file_num))
    while img is not None:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
        ret, thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        erode = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10))
        dilate = cv2.morphologyEx(erode, cv2.MORPH_CLOSE, kernel)

        cv2.imwrite('letters/Boxed_ROIserode' + str(i) + '.jpg', erode)
        cv2.imwrite('letters/Boxed_ROIsthresh' + str(i) + '.jpg', thresh)
        contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        contours, boxes = sort_x(contours)

        os.mkdir(cwd + f'/squared/{file_num:04}')
        for cnt in contours:
            [x, y, w, h] = cv2.boundingRect(cnt)
            if w > 20 or h > 20:
                roi = img[y:y + h, x:x + w]
                square = square_image(roi)
                cv2.imwrite('letters/Boxed_ROIs' + str(i) + '.jpg', roi)
                square.save(f'squared/{file_num:04}/{i:04}.jpg')
                i += 1

        file_num += 1
        img = cv2.imread(image_path + '\Boxed_ROIs{}.jpg'.format(file_num))


def show_img(img):
    cv2.imshow("test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def sort_y(contours):
    boxes = [cv2.boundingRect(c) for c in contours]

    (contours, boxes) = zip(*sorted(zip(contours, boxes), key=lambda b: b[1][1], reverse=False))

    return (contours, boxes)


def sort_x(contours):
    boxes = [cv2.boundingRect(c) for c in contours]
    (contours, boxes) = zip(*sorted(zip(contours, boxes), key=lambda b: b[1][0], reverse=False))
    return (contours, boxes)


def square_image(img):
    img = Image.fromarray(img)
    width, height = img.size
    mx = max(width, height)
    mx *= 2
    new_width = width + (mx - width)
    new_height = height + (mx - height)

    result = Image.new(img.mode, (new_width, new_height), (255, 255, 255))
    result.paste(img, ((mx - height) // 2, (mx - width) // 2))
    result = result.resize((128, 128), resample=Image.Resampling.BICUBIC)
    return result
