import keras.callbacks
import tensorflow as tf
from pnet import get_pnetcls
from keras.optimizer_v1 import SGD
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from utility import *
from sklearn.preprocessing import LabelBinarizer
import pickle
import re
from tensorflow.keras import layers, models
import tensorflow as tf
from keras import backend as K
from verySimpleModel import *
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from pathlib import Path

def append_history(losses, val_losses, accuracy, val_accuracy, history):
    losses = losses + history.history["loss"]
    val_losses = val_losses + history.history["val_loss"]
    accuracy = accuracy + history.history["accuracy"]
    val_accuracy = val_accuracy + history.history["val_accuracy"]
    return losses, val_losses, accuracy, val_accuracy

def train_whole_dataset(patch_dir, model_filepath, train_metadata_filepath):


    #tf.debugging.set_log_device_placement(True)
    '''
    RUN on GPU NOT WORKING
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=2048)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    '''


    unfiltered_filelist = getAllFiles(patch_dir)
    vessel_list = [item for item in unfiltered_filelist]
    train_X = []
    train_y = []
    for el, en in enumerate(vessel_list):
        train_X.append(load_to_numpy(en))
        if "no_vessel" in en:
            train_y.append(0)
        else:
            train_y.append(1)
    lb = LabelBinarizer()
    train_y = lb.fit_transform(train_y)
    train_X = np.array(train_X, dtype=np.float64)
    print('Shape of train_X, train_y: ',train_X.shape, len(train_y))
    # normalize training set
    mean1 = np.mean(train_X)  # mean for data centering
    std1 = np.std(train_X)  # std for data normalization
    train_X -= mean1
    train_X /= std1
    # CREATING MODEL
    patch_size = 32

    model = get_pnetcls(patch_size)
    '''
    DATA AUGMENTATION NOT WORKING
    # Create a data augmentation stage with horizontal flipping, rotations, zooms
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )
    train_X = tf.data.Dataset.from_tensor_slices(train_X, train_y)
    train_X = train_X.batch(16).map(lambda x, y: (data_augmentation(x), y))
    '''
    train_X, test_X, train_y, test_y = train_test_split(train_X, train_y, test_size=0.2, shuffle=True)

    with tf.device('/device:CPU:0'):
        # train model
        print('Training model...')

        #CPU
        # model.fit(train_X, train_y, epochs=10)

        #GPU
        model.fit(train_X, train_y, validation_split=0.2, epochs=20, batch_size=32)
        # saving model
        print('Saving model to ', model_filepath)
        model.save(model_filepath)
        # Create folders if they don't exist already
        Path('train/patched_images/').mkdir(parents=True, exist_ok=True)

        with tf.device("/device:CPU:0"):
            predictions = model.predict(test_X)
        rounded = np.where(np.greater(predictions, 0.5), 1, 0)

        def accuracy(y_pred, y):
            return np.sum(y_pred == y)/len(y)

        accuracy_test = accuracy(y_pred=rounded, y=test_y)
        f1_score_test = f1_score(y_true=test_y, y_pred=rounded)

        print(f"Accuracy on test: {accuracy_test}")
        print(f"f1 on test: {f1_score_test}")


        # saving mean and std
        print('Saving params to ', train_metadata_filepath)
        results = {'mean_train': mean1, 'std_train': std1}

        # Create folder if they don't exist already
        Path(train_metadata_filepath).mkdir(parents=True, exist_ok=True)

        with open(f"{train_metadata_filepath}/result.pkl", 'wb') as handle:
            pickle.dump(results, handle)
        print()
        print('DONE')


#Decide if to train only the classifier (I know the labels and just see the accuracy
#of the classifier or do the classifiationa and wait for the segmentation)
def teach_model(patch_dir, model_filepath, train_metadata_filepath, num_iteration=5):

    #loading labelled patches
    unfiltered_filelist = getAllFiles(patch_dir)
    vessel_list = [item for item in unfiltered_filelist]
    X = []
    y = []
    for el, en in enumerate(vessel_list):
        X.append(load_to_numpy(en))
        if "no_vessel" in en:
            y.append(0)
        else:
            y.append(1)
    lb = LabelBinarizer()
    y = lb.fit_transform(y)
    X = np.array(X, dtype=np.float64)
    print('Shape of X, y: ', X.shape, len(y))

    # TODO: normalize only on training data
    # normalize training set
    mean1 = np.mean(X)  # mean for data centering
    std1 = np.std(X)  # std for data normalization
    X -= mean1
    X /= std1
    # CREATING MODEL
    patch_size = 32

    '''
    def get_n_indices(percentage):
        #extract a small dataset to give initial training
        initial_size = int(percentage*len(train_X))
        return np.random.randint(0, len(train_X), size=initial_size)
    '''

    # TODO: set as a parameter
    X, real_test_X, y, real_test_y = train_test_split(X, y, test_size=0.2, random_state=42)
    percentage_initial_training = 0.2
    train_X ,test_X, train_y, test_y = train_test_split(X,y, train_size=percentage_initial_training, random_state=42)
    #~

    losses, val_losses, accuracies, val_accuracies = [], [], [], []

    model = get_pnetcls(patch_size)
    checkpoint = keras.callbacks.ModelCheckpoint(
        "Active_learning_model", save_best_only=True, verbose=1)
    early_stopping = keras.callbacks.EarlyStopping(patience=10, verbose=1)

    print("Starting to train... ")
    with tf.device("/device:CPU:0"):
        history = model.fit(
            train_X,
            train_y,
            batch_size=32,
            validation_split=0.2,
            epochs=20,
            callbacks=[
                checkpoint,
                keras.callbacks.EarlyStopping(patience=4, verbose=3),
            ],
        )
    losses, val_losses, accuracies, val_accuracies = append_history(
        losses, val_losses, accuracies, val_accuracies, history
    )

    #I should define the number of iteration I want to apply in which I ask for a percentage of images for which I'm uncertain
    for iteration in range(num_iteration):
        with tf.device("/device:CPU:0"):
            test_y_pred = model.predict(test_X)
            real_test_y_pred = model.predict(real_test_X)
        test_y_pred_rounded = np.where(np.greater(test_y_pred, 0.5), 1, 0)

        real_test_y_pred_rounded = np.where(np.greater(real_test_y_pred, 0.5), 1, 0)

        # TODO: put in util (or directly use sklearn function)
        def accuracy(y_pred, y):
            return np.sum(y_pred == y)/len(y)

        accuracy_test = accuracy(y_pred=test_y_pred_rounded, y=test_y)
        f1_score_test = f1_score(y_pred=test_y_pred_rounded, y_true=test_y)

        print(f"Accuracy on test: {accuracy_test}")
        print(f"f1 on test: {f1_score_test}")

        real_accuracy_test = accuracy(y_pred=real_test_y_pred_rounded, y=real_test_y)
        real_f1_score_test = f1_score(y_pred=real_test_y_pred_rounded, y_true=real_test_y)

        print(f"Real Accuracy on test: {real_accuracy_test}")
        print(f"Real f1 on test: {real_f1_score_test}")

        #Make the magic, we should take the images to label essentially

        # To get most uncertain prediction -> argmin(abs(pred - 0.5))

        # Uncertain values with threshold
        # uncertain_values = (np.abs(predictions - 0.5) <= 0.25)


        # Uncertain values count fixed
        count_uncertain_values = 50 # TODO: to put as a parameter

        # TODO: check if axis=0 is correct
        # np.abs(y - 0.5) is smaller the more y is closer to 0.5 (0.5 middle value between 0 and 1)
        most_uncertain_indeces = np.argsort(np.abs(test_y_pred - 0.5), axis=0)
        most_uncertain_indeces = most_uncertain_indeces[:count_uncertain_values].flatten()

        # Works until here

        print(f"train_X.shape: {train_X.shape}")
        print(f"test_X[most_uncertain_indeces, :, :, :].shape: {test_X[most_uncertain_indeces, :, :, :].shape}")

        # Get most uncertain values from test and add them into the train
        train_X = np.vstack((train_X, test_X[most_uncertain_indeces, :, :, :]))
        train_y = np.vstack((train_y, test_y[most_uncertain_indeces, :]))

        # remove most uncertain values from test
        test_X = np.delete(test_X, most_uncertain_indeces, axis=0)
        test_y = np.delete(test_y, most_uncertain_indeces, axis=0)


        #Then I compile again and train again the model
        model.compile(optimizer="SGD",
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        with tf.device("/device:CPU:0"):
            history = model.fit(
                train_X,
                train_y,
                validation_split=0.2,
                epochs=20,
                callbacks=[
                    checkpoint,
                    keras.callbacks.EarlyStopping(patience=10, verbose=1),
                ],
            )

        losses, val_losses, accuracies, val_accuracies = append_history(
            losses, val_losses, accuracies, val_accuracies, history
        )

    #Loading the best model from the training loop
    model = keras.models.load_model("Active_learning_model")

    print("Test set evaluation: ", model.evaluate(train_X, verbose=0, return_dict=True),
          )
    return model

