import os
import numpy as np
import json
import tensorflow as tf

import keras.backend as k
from keras.layers import Dense, Activation
from keras.layers import InputLayer, Input
from keras.models import Sequential, Model, load_model
from keras.layers import concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint


def read_annotation_file(annotation_file_path):
    """
    Read all the annotation information from annotated json file
    :param annotation_file_path: dir which contains all annotated json file
    :return: final_annotation: a dictionary contains all information
    """
    final_annotation_dict = {}

    files = os.listdir(annotation_file_path)
    for file_name in files:
        if file_name.endswith("annotation.json"):
            with open(os.path.join(annotation_file_path,file_name))as f:
                data = json.load(f)
                print(len(data))
                final_annotation_dict = dict(final_annotation_dict.items() + data.items())
    return final_annotation_dict


def load_data_new(data_dict, img_width, img_height):
    data_array = []
    output_array = []
    image = []
    for data in data_dict.values():
        width = float(data['width']) / img_width
        height = float(data['height']) / img_height
        diagonal = np.sqrt(np.square(width) + np.square(height))
        class_h = float(data["size_h"])
        class_w = float(data["size_w"])
        class_d = float(data["size_d"])
        current_object = [1 / width, 1 / height, 1 / diagonal, class_h, class_w, class_d]
        data_array.append(current_object)
        output_array.append(data["distance"])
        image.append(data['name'])
    return np.array(data_array), np.array(output_array), np.array(image)


def train_val_test_split(X, y, image_name, size_train_factor=0.8, size_val_factor=0.1):
    train_length = int(len(X) * size_train_factor)
    val_length = int(len(X) * size_val_factor)
    index = np.random.permutation(np.arange(len(X)))
    X_shuffle = X[index]
    y_shuffle = y[index]
    name_shuffle = image_name[index]
    X_train = X_shuffle[:train_length]
    y_train = y_shuffle[:train_length]
    train_name = name_shuffle[:train_length]
    X_val = X_shuffle[train_length:train_length + val_length]
    y_val = y_shuffle[train_length:train_length + val_length]
    val_name = name_shuffle[train_length:train_length + val_length]
    X_test = X_shuffle[train_length + val_length:]
    y_test = y_shuffle[train_length + val_length:]
    test_name = name_shuffle[train_length + val_length:]
    return X_train, y_train, X_val, y_val, X_test, y_test, train_name, val_name, test_name


def construct_DisNet_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(6,)))
    model.add(Dense(100, activation="selu"))
    model.add(Dense(100, activation="selu"))
    model.add(Dense(100, activation="selu"))
    model.add(Dense(1, activation="selu"))
    optimizer = Adam(1e-4)
    model.compile(optimizer=optimizer,
                  loss="mean_absolute_error")
    return model


def train_net(continue_training=True):
    final_annotation_dict = read_annotation_file("data/annotation/")
    input_data, output_data, image_name = load_data_new(final_annotation_dict, 2592, 1944)
    X_train, \
    y_train, \
    X_val, \
    y_val, \
    X_test, \
    y_test, \
    train_name, \
    val_name, \
    test_name = train_val_test_split(input_data,
                                     output_data,
                                     image_name)

    print("Train data shape: {}".format(X_train.shape))
    print("Validation data shape: {}".format(X_val.shape))
    print("Test data shape: {}".format(X_test.shape))

    DisNet_model_dir = "DisNet_Model"
    DisNet_checkpoints = os.path.join(DisNet_model_dir, "best_disnet_model.keras")
    if not os.path.exists(DisNet_model_dir):
        os.makedirs(DisNet_model_dir)

    print("*******************************************************************************************")
    if os.path.exists(DisNet_checkpoints):
        print("Continue training from checkpoints ...")
        model = load_model(DisNet_checkpoints)
    else:
        print("No model checkpoints founded, construct new model...")
        model = construct_DisNet_model()
    print("*******************************************************************************************")
    callbacks = [EarlyStopping(monitor='val_loss', patience=200, verbose=1),
                 ModelCheckpoint(filepath=DisNet_checkpoints, verbose=1, save_best_only=True)]

    history = model.fit(x=X_train,
                        y=y_train,
                        epochs=10,
                        callbacks=callbacks,
                        verbose=1,
                        batch_size=50,
                        validation_data=(X_val, y_val))
    return history

if __name__ == '__main__':
    train_net(continue_training=True)
