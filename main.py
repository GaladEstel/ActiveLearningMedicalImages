
from learnerClassifier import *
import numpy as np
import tensorflow as tf
from utility import *

def main():
    train_input_path = "train/patched_images/"
    trained_model_path ="model"
    trained_model_path_active = "modelActive"
    results_whole_dataset = "./train/model/results.pkl"
    results_active = "./train/model/resultsActive.pkl"

    train_whole_dataset(train_input_path, trained_model_path, results_whole_dataset)
    #teach_model(train_input_path, trained_model_path_active, results_active, num_iteration=10)

    print("End")

if __name__ == "__main__":
    main()
