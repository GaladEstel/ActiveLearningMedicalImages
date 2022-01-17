import numpy as np
import tensorflow as tf
from utility import *
import train

def main():
    train_images_path = "train/images"
    train_input_path = "train/patched_images/"
    train_CHASE_DB1_path = "CHASE_DB1/patched_images"
    test_path = "train/patched_images_test"
    trained_model_path ="model"
    trained_model_path_active = "modelActive"
    results_whole_dataset = "./train/model/results.pkl"
    results_active = "./train/model/resultsActive.pkl"

    # get the labels for segmentation through CNN and KMeans/canny
    labels = train.train_whole_dataset(train_input_path, trained_model_path, results_whole_dataset, train_CHASE_DB1_path, test_path)
    # train.train_active_learning(train_input_path, trained_model_path_active, results_active, num_iterations=10)
    train.segnet(train_images_path, labels)

    print("End")

if __name__ == "__main__":
    main()