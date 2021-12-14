
from learnerClassifier import *
import numpy as np
import tensorflow as tf
from utility import *

def main():
    train_input_path = "train/patched_images/"
    trained_model_path ="model"
    '''
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    '''

    train_whole_dataset(train_input_path, trained_model_path, "./train/model/results.pkl")


    print("End")

if __name__ == "__main__":
    main()
