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

def append_history(losses, val_losses, accuracy, val_accuracy, history):
    losses = losses + history.history["loss"]
    val_losses = val_losses + history.history["val_loss"]
    accuracy = accuracy + history.history["binary_accuracy"]
    val_accuracy = val_accuracy + history.history["val_binary_accuracy"]
    return losses, val_losses, accuracy, val_accuracy

def train_whole_dataset(patch_dir, model_filepath, train_metadata_filepath):

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

    # train model
    print('Training model...')
    model.fit(train_X, train_y, validation_split=0.2, epochs=10, batch_size=64)
    # saving model
    print('Saving model to ', model_filepath)
    model.save(model_filepath)
    # saving mean and std
    print('Saving params to ', train_metadata_filepath)
    results = {'mean_train': mean1,'std_train':std1}
    with open(train_metadata_filepath, 'wb') as handle:
        pickle.dump(results, handle)
    print()
    print('DONE')


#Decide if to train only the classifier (I know the labels and just see the accuracy
#of the classifier or do the classifiationa and wait for the segmentation)
def teach_model(train_dataset, val_dataset, test_dataset, num_iteration, patch_size):
    #create a small dataset
    losses, val_losses, accuracies, val_accuracies = [], [], [], []
    model = get_pnetcls(patch_size)
    checkpoint = keras.callbacks.ModelCheckpoint(
        "Active_learning_model", save_best_only=True, verbose=1)
    early_stopping = keras.callbacks.EarlyStopping(patience=4, verbose=1)

    print("Starting to train... ")

    history = model.fit(
        train_dataset.cache().shuffle(300).batch(1),
        validation_data=val_dataset,
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
        predictions = model.predict(test_dataset)
        rounded = tf.where(tf.greater(predictions, 0.5), 1, 0)
        #Make the magic, we should take the images to label essentially



        #Then I compile again and train again the model
        opt = SGD(lr=0.01, momentum=0.9)
        model.compile(optimizer=opt,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(
            train_dataset.cache.shuffle(300).batch(1),
            validation_data=val_dataset,
            epochs=20,
            callbacks=[
                checkpoint,
                keras.callbacks.EarlyStopping(patience=4, verbose=1),
            ],
        )

        losses, val_losses, accuracies, val_accuracies = append_history(
            losses, val_losses, accuracies, val_accuracies, history
        )

        #Loading the best model from the training loop
        model = keras.models.load_model("Active_learning_model")

    print("Test set evaluation: ", model.evaluate(test_dataset, verbose=0, return_dict=True),
          )
    return model

