import random
import numpy as np
import keras.callbacks
from Nets.pnet import *
from Nets.verySimpleModel import *
from Nets.resnet import *
from Nets.vgg import *
from utility import *
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from clustering import *
from scipy.stats import entropy


def append_history(losses, val_losses, accuracy, val_accuracy, history):
    losses = losses + history.history["loss"]
    val_losses = val_losses + history.history["val_loss"]
    accuracy = accuracy + history.history["accuracy"]
    val_accuracy = val_accuracy + history.history["val_accuracy"]
    return losses, val_losses, accuracy, val_accuracy


def plot_history(losses, val_losses, accuracies, val_accuracies):
    plt.plot(losses)
    plt.plot(val_losses)
    plt.legend(["train_loss", "val_loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    plt.plot(accuracies)
    plt.plot(val_accuracies)
    plt.legend(["train_accuracy", "val_accuracy"])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()


def get_Xy(path, external_dataset):
    unfiltered_filelist = getAllFiles(path)
    vessel_list = [item for item in unfiltered_filelist]
    X = []
    y = []
    for el, en in enumerate(vessel_list):
        if external_dataset and not "no_vessel" in en: continue
        # X.append(cv2.medianBlur(load_to_numpy(en),5))
        X.append(load_to_numpy(en))
        if "no_vessel" in en:
            y.append(0)
        else:
            y.append(1)
    lb = LabelBinarizer()
    X = np.array(X, dtype=np.float64)
    y = lb.fit_transform(y)
    return X, y


# use it if you want to have a balanced (but smaller) training set and test set (test set to fix; it should remain unbalanced)
def random_under_sampling(X, y):
    indices = np.array(range(len(y)))
    # non vessels are at the beginning
    indices_non_vessels = np.array(range(len(y[y==0])))
    indices_vessels = np.random.choice(indices, size=len(y[y==0]))
    indices_under_sampling = indices_non_vessels.tolist() + indices_vessels.tolist()
    return X[indices_under_sampling], y[indices_under_sampling]

# use it to enlarge the dataset (in particular, the smallest class) by copying samples of that class
def random_over_sampling(X, y):
    y_vessels = y[y==1]
    y_non_vessels = y[y==0]
    difference = len(y_vessels) - len(y_non_vessels)
    X_non_vessels = X[np.where(y==0)[0]]
    X_to_add = X_non_vessels
    for i in range(int(np.floor(difference / len(X_non_vessels)) - 1)):
        X_to_add = np.concatenate((X_to_add, X_non_vessels))
    len_remaining_samples_to_add = difference - len(X_to_add)
    index_remaining_samples_to_add = np.random.choice(X_non_vessels.shape[0], len_remaining_samples_to_add)
    X_to_add = np.concatenate((X_to_add, X_non_vessels[index_remaining_samples_to_add]))
    y_to_add = np.array([0] * len(X_to_add))
    X_over_sampled = np.concatenate((X,X_to_add))
    y_over_sampled = np.concatenate((y,y_to_add.reshape(len(y_to_add),1)))
    return X_over_sampled, y_over_sampled

def normalize(X):
    train_mean = np.mean(X)  # mean for data centering
    train_std = np.std(X)  # std for data normalization
    X -= train_mean
    X /= train_std
    return X, train_mean, train_std


def shuffle_data(X,y):
    indices = np.array(range(len(y)))
    random.shuffle(indices)
    return X[indices], y[indices]


def train_whole_dataset(patch_dir, model_filepath, train_metadata_filepath, path_CD, test_path):
    train_size = 0.8
    np.random.seed(42)
    # X, y origin dataset
    X, y = get_Xy(patch_dir, external_dataset=False)
    # Add images of external dataset
    # X_CD_no_vessel, y_CD_no_vessel = get_Xy(path_CD, external_dataset=True)
    # X = np.concatenate((X, X_CD_no_vessel))
    # y = np.concatenate((y, y_CD_no_vessel))
    # X, y = random_under_sampling(X, y)
    # X, y = random_over_sampling(X, y)
    X_train, y_train = shuffle_data(X,y)
    X_train, train_mean, train_std = normalize(X_train)
    X_test, y_test = get_Xy(test_path, external_dataset=False)
    X_test -= train_mean
    X_test /= train_std


    # I try KMeans
    # for index,X_train_sample in enumerate(X_train):
    #    kmeans(X_train_sample, y_train[index][0])

    patch_size = 32
    # model = get_very_simple_model(patch_size)
    model = get_pnetcls(patch_size)
    # model = get_resnet(patch_size)
    # model = get_vgg(patch_size)

    with tf.device('/device:CPU:0'):
        # train model
        print('Training model...')
        # define class weights - 0 is non vessels, 1 is vessels
        # weights = {0:8, 1:1}
        history = model.fit(
            X_train,
            y_train,
            epochs=20,
            batch_size=32,
            validation_split=0.2,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5, verbose=1),
                keras.callbacks.ModelCheckpoint(
                    "FullModelCheckpoint.h5", verbose=1, save_best_only=True
                ),
            ],
            # class_weight=weights,
        )

        plot_history(
            history.history["loss"],
            history.history["val_loss"],
            history.history["accuracy"],
            history.history["val_accuracy"],
        )

        with tf.device("/device:CPU:0"):
            y_pred = model.predict(X_test)
        y_pred_rounded = np.where(np.greater(y_pred, 0.5), 1, 0)

        accuracy_score_test = accuracy_score(y_test, y_pred_rounded)
        precision_score_test = precision_score(y_test, y_pred_rounded)
        recall_score_test = recall_score(y_test, y_pred_rounded)
        f1_score_test = f1_score(y_test, y_pred_rounded)

        print(f"Accuracy on test: {accuracy_score_test}")
        print(f"Precision on test: {precision_score_test}")
        print(f"Recall on test: {recall_score_test}")
        print(f"f1 on test: {f1_score_test}")
        print(classification_report(y_test, y_pred_rounded))

        print()
        print('DONE')


def train_active_learning(patch_dir, model_filepath, train_metadata_filepath, num_iterations, metrics="least_confidence"):
    np.random.seed(42)
    X, y = get_Xy(patch_dir)
    # X, y = random_under_sampling(X, y)
    X, y = shuffle_data(X, y)
    # We start with a 20% of the samples
    train_size = 0.2
    X_train, X_test, y_train, y_test = split_and_normalize(X, y, train_size)

    # Creating lists for storing metrics
    losses, val_losses, accuracies, val_accuracies = [], [], [], []

    patch_size = 32
    # model = get_very_simple_model(patch_size)
    # model = get_pnetcls(patch_size)
    # model = get_resnet(patch_size)
    model = get_vgg(patch_size)

    print("Starting to train... ")
    with tf.device("/device:CPU:0"):
        history = model.fit(
            X_train,
            y_train,
            epochs=10,
            batch_size=32,
            validation_split=0.2,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5, verbose=1),
                keras.callbacks.ModelCheckpoint(
                    "ALModelCheckpoint.h5", verbose=1, save_best_only=True
                ),
            ],
        )

    losses, val_losses, accuracies, val_accuracies = append_history(
        losses, val_losses, accuracies, val_accuracies, history
    )

    plot_history(
        history.history["loss"],
        history.history["val_loss"],
        history.history["accuracy"],
        history.history["val_accuracy"],
    )

    #  Active Learning iterations
    for iteration in range(num_iterations):
        with tf.device("/device:CPU:0"):
            y_pred = model.predict(X_test)
        y_pred_rounded = np.where(np.greater(y_pred, 0.5), 1, 0)

        accuracy_score_test = accuracy_score(y_test, y_pred_rounded)
        precision_score_test = precision_score(y_test, y_pred_rounded)
        recall_score_test = recall_score(y_test, y_pred_rounded)
        f1_score_test = f1_score(y_test, y_pred_rounded)

        print(f"Accuracy on test: {accuracy_score_test}")
        print(f"Precision on test: {precision_score_test}")
        print(f"Recall on test: {recall_score_test}")
        print(f"f1 on test: {f1_score_test}")
        print(classification_report(y_test, y_pred_rounded))

        # Uncertain values count fixed
        count_uncertain_values = 50  # TODO: to put as a parameter

        if metrics == "least_confidence":
            # TODO: check if axis=0 is correct
            # np.abs(y - 0.5) is smaller the more y is closer to 0.5 (0.5 middle value between 0 and 1)
            most_uncertain_indeces = np.argsort(np.abs(y_pred - 0.5), axis=0)
            most_uncertain_indeces = most_uncertain_indeces[:count_uncertain_values].flatten()

        elif metrics == "entropy":
            entropy_y = np.transpose(entropy(np.transpose(y_pred)))
            most_uncertain_indeces = np.argpartition(-entropy_y, count_uncertain_values - 1, axis=0)[
                                     :count_uncertain_values]

        print(f"X_train.shape: {X_train.shape}")
        print(f"X_test[most_uncertain_indeces, :, :, :].shape: {X_test[most_uncertain_indeces, :, :, :].shape}")

        # Get most uncertain values from test and add them into the train
        X_train = np.vstack((X_train, X_test[most_uncertain_indeces, :, :, :]))
        y_train = np.vstack((y_train, y_test[most_uncertain_indeces, :]))

        # remove most uncertain values from test
        X_test = np.delete(X_test, most_uncertain_indeces, axis=0)
        y_test = np.delete(y_test, most_uncertain_indeces, axis=0)

        # Then I compile again and train again the model
        model = get_very_simple_model(patch_size)
        model.compile(optimizer="SGD",
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        with tf.device("/device:CPU:0"):
            history = model.fit(
                X_train,
                y_train,
                epochs=10,
                batch_size=32,  # it was missing in the first version
                validation_split=0.2,
                callbacks=[
                    keras.callbacks.EarlyStopping(patience=5, verbose=1),
                    keras.callbacks.ModelCheckpoint(
                        "ALModelCheckpoint.h5", verbose=1, save_best_only=True
                    ),
                ],
            )

        losses, val_losses, accuracies, val_accuracies = append_history(
            losses, val_losses, accuracies, val_accuracies, history
        )

    # End of AL iterations

    # Loading the best model from the training loop
    model = keras.models.load_model("ALModelCheckpoint")

    print("Test set evaluation: ", model.evaluate(X_train, verbose=0, return_dict=True))
    return model