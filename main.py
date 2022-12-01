"""
Author: Caleb Ehrisman
Course- CSC-514 Computer Vision
Assignment - Semester Project - Text Transcription from images
"""

import argparse
import os
import sys

"""
    parse gathers command line arguments.

    :return: a list of all parsed arguments
"""


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Path for image file to load')
    return parser.parse_args()


def main():
    check_args()


def check_args():
    args = parse()
    path = None

    if args.image is not None:
        path = args.image

    if path is None:
        sys.exit("No file path entered.")

    if not os.path.exists(path):
        sys.exit(path + " File not found. Check file path name")


if __name__ == "__main__":
    main()

    # Error checking for path input and checking if file exists.
    # If here then file exists
