import keras.callbacks
import tensorflow as tf
from pnet import get_pnetcls
from keras.optimizer_v1 import SGD

def append_history(losses, val_losses, accuracy, val_accuracy, history):
    losses = losses + history.history["loss"]
    val_losses = val_losses + history.history["val_loss"]
    accuracy = accuracy + history.history["binary_accuracy"]
    val_accuracy = val_accuracy + history.history["val_binary_accuracy"]
    return losses, val_losses, accuracy, val_accuracy

def teach_model(train_dataset, val_dataset, test_dataset, num_iteration, patch_size):
    #create a small dataset
    losses, val_losses, accuracies, val_accuracies = [], [], [], []
    model = get_pnetcls(patch_size)
    checkpoint = keras.callbacks.ModelCheckpoint(
        "Active_learning_model", save_best_only=True, verbose=1)
    early_stopping = keras.callbacks.EarlyStopping(patience=4, verbose=1)

    print("Starting to train")

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

