#Preprocessing of the eye, applying the mask
import argparse
import sys
import os
import numpy as np
import re
import pandas as pd
from PIL import Image

def applyMask(input, mask):
    masked = input
    masked[np.where(mask == 0)] = 0
    return masked

def createAndSaveImage(image_to_save, output_name):
    savingImage = Image.fromarray(image_to_save, "RGB")
    savingImage.save(output_name)
    return True
#TODO add args
def main():
    #Load training and test data
    train_path = "train/images/"
    mask_train_path = "train/mask/"
    test_path = "test/images/"
    mask_test_path = "test/mask/"
    masked_train_path = "train/masked_train/"
    masked_test_path = "test/masked_test/"

    input_train_images = [item for item in os.listdir(train_path) if re.search("_training", item)]
    mask_train_images = [item for item in os.listdir(mask_train_path) if re.search("_training_mask", item)]
    input_test_images = [item for item in os.listdir(test_path) if re.search("_test", item)]
    mask_test_images = [item for item in os.listdir(mask_test_path) if re.search("_test_mask", item)]

    print("Loading and saving... ")
    for el, en in zip(input_train_images, mask_train_images):
        createAndSaveImage(applyMask(np.array(Image.open(train_path + el)), np.array(Image.open(mask_train_path + en))), masked_train_path + el)

    for el, en in zip(input_test_images, mask_test_images):
        createAndSaveImage(applyMask(np.array(Image.open(test_path + el)), np.array(Image.open(mask_test_path + en))), masked_test_path + el)

    print("Done")

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_data_dir", type=str,
                        help='data dictionary.')
    parser.add_argument("--target_dir", type=str, help='Directory for saving the images after the skull stripping process.')
    return parser.parse_args(argv)


#TODO add args
if __name__ == '__main__':
    main()