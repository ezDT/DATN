import ast
import cv2
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import sys
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, Input, Lambda, MaxPooling2D, ReLU, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model
import time

def create_model(chosen_model):
    model_input = Input(shape=(360, 640) + (3,))

    # Number of output values: 
    #       5 values of percentage for each class
    #       ~ [Rat vang - 0, Vang - 1, Trung binh - 2, Dong - 3, Tac - 4]

    n_outputs = 5
    
    name = 'traffic_status'
    
    if chosen_model == 'AlexNet':
        name += '_mini_AlexNet'
        
        x = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', name="conv1", activation="relu")(model_input)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="pool1")(x)
        x = BatchNormalization()(x)

        x = Conv2D(filters=96, kernel_size=(5, 5), strides=(1, 1), padding='same', name="conv2", activation="relu")(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="pool2")(x)
        x = BatchNormalization()(x)

        x = Flatten()(x)
        
        x = Dense(384, activation='relu', name="fc3")(x)
        x = Dropout(0.5, name="dropout3")(x)
        x = Dense(192, activation='relu', name="fc4")(x)
        x = Dropout(0.5, name="dropout4")(x)
        x = Dense(n_outputs, activation='softmax', name="fc5")(x)

        base_model = Model(inputs=model_input, outputs=x)
    
    elif chosen_model == 'LeNet5':
        name += '_modified_LeNet5'
        
        x = Conv2D(filters=32, kernel_size=(5, 5), name="conv1", activation="relu")(model_input)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name="pool1")(x)
        
        x = Conv2D(filters=48, kernel_size=(5, 5), name="conv2", activation="relu")(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name="pool2")(x)
        
        x = Flatten()(x)
        
        x = Dense(192, name="fc3")(x)
        x = Dropout(0.6, name="dropout3")(x)
        
        x = Dense(96, name="fc4")(x)
        x = Dropout(0.6, name="dropout4")(x)
        
        x = Flatten()(x)
        x = Dense(n_outputs, activation='softmax', name="fc5")(x)

        base_model = Model(inputs=model_input, outputs=x)
        
    else:
        if chosen_model == 'InceptionV3':
            name += '_InceptionV3'
            pretrained_model = InceptionV3(weights='imagenet', include_top=False)
            
        if chosen_model == 'MobileNetV2':
            name += '_MobileNetV2'
            pretrained_model = MobileNetV2(weights='imagenet', include_top=False)
        
        if chosen_model == 'ResNet50':
            name += '_ResNet50'
            pretrained_model = ResNet50(weights='imagenet', include_top=False)

        pretrained_model.trainable = False

        x = pretrained_model(pretrained_model.output)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        x = Dense(n_outputs)(x)

        base_model = Model(inputs=model_input, outputs=x)
        
    return base_model, name

# %% Main function

if __name__ == "__main__":
    # Choose a model's name
    # Available models: ['AlexNet', 'LeNet5', 'InceptionV3', 'MobileNetV2', 'ResNet50']
    
    avail_model = ['AlexNet', 'LeNet5', 'InceptionV3', 'MobileNetV2', 'ResNet50']
    
    # Landmark detection 
    result = []

    image = cv2.imread('F:/Bip/DATN/experiment/dataset_test/original/3 trung binh/20230402_144533_000322.jpg')
    
    image = cv2.resize(image, (640, 360), interpolation = cv2.INTER_AREA)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Model selection with an iterator
    it = 0
    model, saved_name = create_model(avail_model[it])
        
    # Load the saved model - use ONLY when inference (not to re-train again/call specific saved model)
    #model = tf.keras.models.load_model("traffic_status_mini_AlexNet.h5")
    model.load_weights("traffic_status_mini_AlexNet.h5")
    
    # Predict for 2D input image
    X_test = []
    X_test.append(image_rgb)
    X_test = np.array(X_test)
    
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()
    
    print(y_pred[0])
    
    print(f"Traffic status inference time: {(end_time-start_time):.3f}s")
    
    cv2.waitKey(0)