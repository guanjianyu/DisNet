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
from keras.models import model_from_json
from DataGenerate import read_annotation_files, load_data_new, train_val_test_split


class DistanceNet(object):

    def __init__(self):
        self.learning_rate = 1e-4
        self.weight_path = "Best_Model.h5"
        self.model_config = "Model_config.json"
        self.model_save_path = "DisNet_Model.keras"
        self.n_epochs = 100
        self.epoch_without_improving = 100
        self.batch_size = 50
        self.model = self.distance_model()

    def distance_model(self):
        dis_model = Sequential()
        dis_model.add(InputLayer(input_shape=(6,)))
        dis_model.add(Dense(100, activation="selu"))
        dis_model.add(Dense(100, activation="selu"))
        dis_model.add(Dense(1, activation="selu"))
        return dis_model

    def train(self, X_train, y_train, X_val, y_val):
        optimizer = Adam(self.learning_rate)
        self.model.compile(optimizer=optimizer, loss="mean_absolute_error")
        callbacks = [EarlyStopping(monitor='val_loss', patience=self.epoch_without_improving, verbose=1),
                     ModelCheckpoint(filepath=self.weight_path, verbose=1, save_best_only=True)]
        self.model.fit(x=X_train,
                       y=y_train,
                       epochs=self.n_epochs,
                       callbacks=callbacks,
                       verbose=1,
                       batch_size=self.batch_size,
                       validation_data=(X_val, y_val))
        self.model.load_weights(filepath=self.weight_path)
        return self.model

    def evaluate(self, eva_model, X_val, y_val):
        result = eva_model.evaluate(x=X_val, y=y_val)
        for name, value in zip(eva_model.metrics, result):
            print(name, value)

    def save_model(self):
        self.model.save(self.model_save_path)
        config = self.model.to_json()
        with open(self.model_config, "w") as json_file:
            json_file.write(config)


def train_model(filepath, image_width, image_height):
    annotation_file = read_annotation_files(filepath)
    input_data, output_data, image_name = load_data_new(annotation_file, image_width, image_height)
    X_train, y_train, X_val, y_val, X_test, y_test, train_name, val_name, test_name = train_val_test_split(input_data,
                                                                                                           output_data,
                                                                                                           image_name)

    model = DistanceNet()
    result_model = model.train(X_train, y_train, X_val, y_val)
    model.save_model()


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train DisNet to estimate distance")

    parser.add_argument("command", metavar="<command>", help="train")
    parser.add_argument("--filepath", required=True, metavar="path/to/annotation_file")
    parser.add_argument("--image_width", type=int, help="Image width in pixel", required=True)
    parser.add_argument("--image_height", type=int, help="image height in pixel", required=True)
    args = parser.parse_args()

    if args.command == "train":
        train_model(args.filepath,args.image_width,args.image_height)
        print("zasdsa")
