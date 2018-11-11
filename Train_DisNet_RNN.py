import os
import numpy as np
import json
import tensorflow as tf

import keras.backend as k
from keras.layers import *
from keras.layers import InputLayer, Input
from keras.models import Sequential, Model, load_model
from keras.layers import concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedShuffleSplit


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
            order_data = []
            with open(annotation_file_path + file_name)as f:
                data = json.load(f)
                order_keys = np.sort(data.keys())
                for key in order_keys:
                    order_data.append(data[key])
                    final_annotation_dict[file_name] = order_data
    return final_annotation_dict


def data_generation_for_each_frame(final_annotation_dict, img_width, img_height):
    """
    Generate usual information for each frame and save all info from one annotation file as an entity of a dictionary
    :param final_annotation_dict:a dictionary contains all information
    :param img_width:
    :param img_height:
    """
    data_dict = {}
    for key in final_annotation_dict.keys():
        data_array = []
        datas = final_annotation_dict[key]
        print(key)
        print(len(datas))
        for data in datas:
            width = float(data['width']) / img_width
            height = float(data['height']) / img_height
            top = (float(data['y']) / img_height)
            left = (float(data['x']) / img_width)
            bottom = float(data['y'] + data['height']) / img_height
            right = float(data['x'] + data['width']) / img_width
            diagonal = np.sqrt(np.square(width) + np.square(height))
            class_h = float(data["size_h"])
            class_w = float(data["size_w"])
            class_d = float(data["size_d"])
            distance = float(data["distance"])
            current_data = [class_h, class_w, class_d, 1 / width, 1 / height, 1 / diagonal, top, left, bottom, right,
                            distance]

            data_array.append(current_data)
        data_dict[key] = data_array
    return data_dict


def sequence_data_preparation_three(data_set):
    """
    generate sequence data for training DisNet RNN network
    :param data_set:
    :return:
    """
    data_array = []
    data_array_one = []
    data_array_two = []
    data_array_three = []
    data_array_reverse = []
    for data in data_set.values():
        print(len(data))
        set_len = len(data)
        index = 0
        zero_list = [i * 0 for i in range(11)]
        while index + 3 < set_len:
            data_one = [zero_list, zero_list, data[index - 1], data[index]]
            data_two = [zero_list, data[index - 2], data[index - 1], data[index]]
            data_three = [data[index - 3], data[index - 2], data[index - 1], data[index]]
            data_array.append(data_one)
            data_array.append(data_two)
            data_array.append(data_three)
            data_array_one.append(data_one)
            data_array_two.append(data_two)
            data_array_three.append(data_three)
            index += 1
        index_reverse = set_len - 1
        while index_reverse - 2 >= 0:
            data_one = [zero_list, zero_list, data[index_reverse], data[index_reverse - 1]]
            data_two = [zero_list, data[index_reverse], data[index_reverse - 1], data[index_reverse - 2]]
            data_three = [data[index_reverse], data[index_reverse - 1], data[index_reverse - 2],
                          data[index_reverse - 3]]

            data_array_reverse.append(data_one)
            data_array_reverse.append(data_two)
            data_array_reverse.append(data_three)
            data_array_two.append(data_two)
            data_array_three.append(data_three)
            index_reverse = index_reverse - 1
    return np.concatenate((np.array(data_array), np.array(data_array_reverse))), np.array(data_array_one), np.array(
        data_array_two), np.array(data_array_three)


def input_output_generation(data):
    """
    Split sequence data into X,y for distance estimation and position prediction
    :param data: sequence data
    :return:
    """
    x_data_distance = data[:, :-1, :6]
    x_data_position = data[:, :-1, 6:-1]
    y_data_predict = data[:, 1:, 6:-1]
    y_data_distance = data[:, :-1, -1].reshape(-1, 3, 1)
    return x_data_distance, x_data_position, y_data_predict, y_data_distance


def get_train_valid_test_data(data_sequence, DisNet_dataDict_save_path):
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=86)
    for train_val_index, test_index in split.split(data_sequence, data_sequence[:, 0, 0]):
        train_val_set = data_sequence[train_val_index]
        test_set = data_sequence[test_index]

    for train_index, val_index in split.split(train_val_set, train_val_set[:, 0, 0]):
        train_set = train_val_set[train_index]
        val_set = train_val_set[val_index]
    X_train_distance, X_train_position, y_train_pred, y_train_dist = input_output_generation(train_set)
    X_val_distance, X_val_positioin, y_val_pred, y_val_dist = input_output_generation(val_set)
    X_test_distance, X_test_position, y_test_pred, y_test_dist = input_output_generation(test_set)

    standscale = StandardScaler()
    X_train_distance_scale_reshape = standscale.fit_transform(X_train_distance.reshape(-1, 6))
    X_train_distance_scale = X_train_distance_scale_reshape.reshape(-1, 3, 6)
    X_val_distacne_scale_reshape = standscale.transform(X_val_distance.reshape(-1, 6))
    X_val_distance_scale = X_val_distacne_scale_reshape.reshape(-1, 3, 6)
    X_test_distance_scale_reshape = standscale.transform(X_test_distance.reshape(-1, 6))
    X_test_distance_scale = X_test_distance_scale_reshape.reshape(-1, 3, 6)
    scaler_name = 'DisNet_RNN_Model/model_vg_stand_scaler.sav'
    joblib.dump(standscale, scaler_name)

    DisNet_dataDict = {"X_train_distance": X_train_distance_scale,
                       "X_train_position": X_train_position,
                       "y_train_pred": y_train_pred,
                       "y_train_dist": y_train_dist,
                       "X_val_distance": X_val_distance_scale,
                       "X_val_position": X_val_positioin,
                       "y_val_pred": y_val_pred,
                       "y_val_dist": y_val_dist,
                       "X_test_distance": X_test_distance_scale,
                       "X_test_position": X_test_position,
                       "y_test_pred": y_test_pred,
                       "y_test_dist": y_test_dist}
    np.save(DisNet_dataDict_save_path, DisNet_dataDict)
    return DisNet_dataDict


def load_train_valid_test_data(DisNet_dataDict_save_path):
    """
    load train valid test data from npy file
    :param DisNet_dataDict_save_path: npy file which contains training data
    :return:
    """
    data_dict = np.load(DisNet_dataDict_save_path).item()
    X_train_distance = data_dict["X_train_distance"]
    X_train_position = data_dict["X_train_position"]
    y_train_pred = data_dict["y_train_pred"]
    y_train_dist = data_dict["y_train_dist"]
    X_val_distance = data_dict["X_val_distance"]
    X_val_position = data_dict["X_val_position"]
    y_val_pred = data_dict["y_val_pred"]
    y_val_dist = data_dict["y_val_dist"]
    X_test_distance = data_dict["X_test_distance"]
    X_test_position = data_dict["X_test_position"]
    y_test_pred = data_dict["y_test_pred"]
    y_test_dist = data_dict["y_test_dist"]
    return (X_train_distance, X_train_position, y_train_pred, y_train_dist), \
           (X_val_distance, X_val_position, y_val_pred, y_val_dist), \
           (X_test_distance, X_test_position, y_test_pred, y_test_dist)


def construct_DisNet_RNN():
    """
    construct keras DisNet RNN model
    :return: keras model for distance estimatio,n keras model for position prediction
    """
    inputs_g = Input(shape=(3, 6))
    x_g = GRU(128, return_sequences=True, unroll=True)(inputs_g)
    x_g = GRU(128, return_sequences=True, unroll=True)(x_g)
    x_g = GRU(64, return_sequences=True, unroll=True)(x_g)

    logits_distance_g = Dense(1, activation="relu", name="distance")(x_g)

    model_distance = Model(inputs=inputs_g, outputs=logits_distance_g)
    model_distance.compile(optimizer='adam', loss='mean_squared_error')
    print(model_distance.summary())

    input_1 = Input(shape=(3, 4))
    x = GRU(128, return_sequences=True, unroll=True)(input_1)
    logits = Dense(4, activation='relu')(x)
    model_position = Model(inputs=input_1, outputs=logits)
    model_position.compile(optimizer='adam', loss='mean_squared_error')
    print(model_position.summary())
    return model_distance, model_position


def train_net(DisNet_dataDict_save_path, contrinue_training=True):
    """
    Train Distance model and position model.
    :param DisNet_dataDict_save_path:
    :return:
    """
    if not os.path.exists(DisNet_dataDict_save_path):
        final_annotation_dict = read_annotation_file("data/annotation/")
        data_dict = data_generation_for_each_frame(final_annotation_dict, 2592, 1944)
        data_sequence, data_one, data_two, data_three = sequence_data_preparation_three(data_dict)
        DisNet_dataDict = get_train_valid_test_data(data_sequence,"DisNet_RNN_DataSet_scaled-2018.11.11.npy")

    (X_train_distance, X_train_position, y_train_pred, y_train_dist), \
    (X_val_distance, X_val_position, y_val_pred, y_val_dist), \
    (X_test_distance, X_test_position, y_test_pred, y_test_dist) = load_train_valid_test_data(
        "DisNet_RNN_DataSet_scaled-2018.11.11.npy")

    Model_savedir = "DisNet_RNN_Model"
    if not os.path.exists(Model_savedir):
        os.makedirs(Model_savedir)
    print(X_train_distance.shape)
    print(X_train_position.shape)
    print(y_train_pred.shape)

    model_distance_checkpoints=os.path.join(Model_savedir,"best_distance_model_DisNet_rnn_today.h5")
    model_position_checkpoints=os.path.join(Model_savedir,"best_distance_model_DisNet_rnn.h5")
    print("*******************************************************************************************")
    if os.path.exists(model_distance_checkpoints) and os.path.exists(model_position_checkpoints) and contrinue_training:
        print("Continue training from checkpoints ...")
        model_distance=load_model(model_distance_checkpoints)
        model_position=load_model(model_position_checkpoints)
    else:
        print("No model checkpoints founded, construct new model...")
        model_distance, model_position = construct_DisNet_RNN()
    print("*******************************************************************************************")

    distance_callbacks = [EarlyStopping(monitor='val_loss', patience=50, verbose=1),
                          ModelCheckpoint(filepath=model_distance_checkpoints, verbose=1, save_best_only=True)]

    position_callbacks = [EarlyStopping(monitor='val_loss', patience=50, verbose=1),
                          ModelCheckpoint(filepath=model_position_checkpoints, verbose=1, save_best_only=True)]

    print("Start to train distance model....")
    distance_history = model_distance.fit(x=X_train_distance,
                                          y=y_train_dist,
                                          epochs=20,
                                          verbose=1,
                                          batch_size=150,
                                          callbacks=distance_callbacks,
                                          validation_data=(X_val_distance, y_val_dist)
                                          )
    print("Distance model training finished...")
    print("*******************************************************************************************")

    print("Start to train position model...")
    position_history = model_position.fit(x=X_train_position,
                                          y=y_train_pred,
                                          epochs=20,
                                          verbose=1,
                                          batch_size=150,
                                          callbacks=position_callbacks,
                                          validation_data=(X_val_position, y_val_pred)
                                          )
    return distance_history,position_history

if __name__ == '__main__':
    train_net("DisNet_RNN_DataSet_scaled-2018.11.11.npy")
