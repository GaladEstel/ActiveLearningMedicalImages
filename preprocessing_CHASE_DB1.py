#Preprocessing of the eye, applying the mask
import argparse
import sys
import os
import numpy as np
import re
import pandas as pd
from utility import *

#TODO add args
def main():
    #Load training and test data
    train_path = "CHASE_DB1/images/"
    mask_train_path = "CHASE_DB1/mask/"
    masked_train_path = "CHASE_DB1/masked_images/"

    input_train_images = [item for item in os.listdir(train_path)]
    mask_train_images = [item for item in os.listdir(mask_train_path)]

    print("Loading and saving... ")
    for el, en in zip(input_train_images, mask_train_images):
        createAndSaveImage(applyMask(np.array(Image.open(train_path + el)), np.array(Image.open(mask_train_path + en))), masked_train_path + el)

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