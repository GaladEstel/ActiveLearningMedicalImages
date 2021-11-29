import keras.callbacks

from pnet import get_pnetcls

def teach_model(train_dataset, test_dataset, patch_size):
    #create a small dataset
    model = get_pnetcls(patch_size)
    checkpoint = keras.callbacks.ModelCheckpoint(
        "Active_learning_model", save_best_only=True, verbose=1)
    early_stopping = keras.callbacks.EarlyStopping(patience=4, verbose=1)

    print("Starting to train")

    history = model.fit(
        train_dataset.cache().shuffle()
        #TO BE CONTINUED
    )

    pass
