# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2

from keras.utils.generic_utils import Progbar
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation, Dense, Flatten, Dropout
from keras.layers import Cropping2D, Lambda
from keras.optimizers import Adam
from keras.regularizers import l2

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

np.random.seed(123)
resize_img_shape = (96, 96, 3)


def read_img(paths, resize=True, new_size=resize_img_shape, folder_path=""):
    print("read in images...")
    if resize:
        X = np.zeros([len(paths), *new_size])
    else:
        X = np.zeros([len(paths), 160, 320, 3])
    progbar = Progbar(X.shape[0])  # progress bar for normalization status tracking
    for i in range(len(paths)):
        if resize:
            X[i] = cv2.resize(plt.imread(folder_path + paths[i]), new_size[1::-1])
        else:
            X[i] = plt.imread(folder_path + paths[i])
        progbar.add(1)
    print()
    print("finished")
    return X


def rotation_augmentation(X, angle_range):
    X_rot = np.copy(X)
    angle = np.random.randint(-angle_range, angle_range)
    X_rot = ndimage.rotate(X, angle, reshape=False, order=0)
    return X_rot


def shift_augmentation(X, h_range, w_range):
    X_shift = np.copy(X)
    h_random = (np.random.rand() * 2. - 1.) * h_range
    w_random = (np.random.rand() * 2. - 1.) * w_range
    X_shift = ndimage.shift(X, (h_random, w_random, 0), order=0)
    return X_shift




def get_model():
    model = Sequential()
    model.add(Cropping2D(cropping=((35,16), (0,0)), input_shape=resize_img_shape))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    
    model.add(Convolution2D(32, 5, 5, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 5, 5, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))
    
    
    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    
    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss='mse', metrcs=['mean_absolute_error'])

    return model

 

def generator(X, y, batch_size, prob=None):
    batch_X = np.zeros((batch_size, *resize_img_shape))
    batch_y = np.zeros(batch_size)

    while True:
        for i in range(0, batch_size, 2):
            index= np.random.choice(len(X),1,p=prob)[0]
            temp = rotation_augmentation(X[index], np.pi/36.)
            temp = shift_augmentation(temp, 0.1, 0.1)
            batch_X[i] = temp
            batch_y[i] = y[index]
            
            imgs = cv2.flip(X[index], 1)
            temp = rotation_augmentation(imgs, np.pi/36.)
            temp = shift_augmentation(temp, 0.1, 0.1)
            batch_X[i+1] = cv2.flip(temp[0],1)
            batch_y[i+1] = -batch_y[i]
        yield batch_X, batch_y


if __name__ == '__main__':
    
    data= pd.read_csv("./udacity_data/driving_log.csv")
    data.columns = ["center", "left", "right", "steering", "throttle", "brake", "speed"]
    data = data[["center", "steering"]]
    X_val = read_img(data['center'], resize=True, folder_path="./udacity_data/")
    y_val = data['steering'].values
    
    data= pd.read_csv("./data/driving_log.csv")
    data.columns = ["center", "left", "right", "steering", "throttle", "brake", "speed"]
    data = data[["center", "steering"]]
    data['prob'] = pd.cut(data['steering'], 11)
    prob_inv = 1. / data['prob'].value_counts()
    data['prob'] = data['prob'].astype(str).replace(prob_inv.index, prob_inv)
    prob = data['prob'] / data['prob'].sum()
    X = read_img(data['center'], resize=True)
    y = data['steering'].values
    
    model = get_model() 
    model.fit_generator(generator(X, y, 128, prob), 
                        samples_per_epoch=2*len(X), 
                        nb_epoch=3, 
                        validation_data=(X_val, y_val))
    
    model.save('model.h5')