
from learnerClassifier import *
import numpy as np
import tensorflow as tf
from utility import *

def main():
    train_input_path = "train/patched_images/"
<<<<<<< HEAD
    trained_model_path = "model"
    train_metadata_filepath = "./train/model"
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

    train_whole_dataset(train_input_path, trained_model_path, train_metadata_filepath)
=======
    trained_model_path ="model"
    trained_model_path_active = "modelActive"
    results_whole_dataset = "./train/model/results.pkl"
    results_active = "./train/model/resultsActive.pkl"
>>>>>>> 0e8e304faacb33f81f96930fc984e1541ef9bf8b

    #train_whole_dataset(train_input_path, trained_model_path, results_whole_dataset)
    teach_model(train_input_path, trained_model_path_active, results_active, num_iteration=10)

    print("End")

if __name__ == "__main__":
    main()
