from pnet import get_pnetcls
from learnerClassifier import *
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from utility import *

def main():
    train_input_path = "train/patched_images/"
    trained_model_path ="/train/model/whole_train"
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


    print("Cazzi")
    train_whole_dataset(train_input_path, trained_model_path, "/train/")


    print("End")

if __name__ == "__main__":
    main()
