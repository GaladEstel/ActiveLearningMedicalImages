from pnet import get_pnetcls
from learnerClassifier import *
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory


def main():
    # Load training and validation sets
    ds_train_ = image_dataset_from_directory(
        './train/patched_images',
        labels='inferred',
        label_mode='binary',
        image_size=[32, 32],
        interpolation='nearest',
        batch_size=64,
        shuffle=True,
    )


    print("End")

if __name__ == "__main__":
    main()
