from pnet import get_pnetcls
from learnerClassifier import *
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from utility import *

def main():
    train_input_path = "train/patched_images/"
    trained_model_path ="/train/model/"
    train_whole_dataset(train_input_path, trained_model_path, "/train/")


    print("End")

if __name__ == "__main__":
    main()
