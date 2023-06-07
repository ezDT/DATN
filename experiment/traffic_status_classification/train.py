import ast
import cv2
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re
from sklearn.metrics import accuracy_score
import sys
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, Input, Lambda, MaxPooling2D, ReLU, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical


data_dir = r'F:/Bip/DATN/experiment/dataset_test/all_resized'


# %% Define base model for transfer learning


def create_model(chosen_model):
    model_input = Input(shape=(360, 640) + (3,))

    # Number of output values: 
    #       5 values of percentage for each class
    #       ~ [Rat vang - 0, Vang - 1, Trung binh - 2, Dong - 3, Tac - 4]

    n_outputs = 5
    
    name = 'traffic_status'
    
    if chosen_model == 'AlexNet_simplified':
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
        base_model.load_weights("alexnet_weights.h5", by_name=True)

    elif chosen_model == 'AlexNet_full':
        name += '_original_AlexNet'

        x = Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), name="conv1", activation="relu")(model_input)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="pool1")(x)
        x = BatchNormalization()(x)

        x = ZeroPadding2D((2, 2))(x)
        con2_split1 = Lambda(lambda z: z[:,:,:,:48])(x)
        con2_split2 = Lambda(lambda z: z[:,:,:,48:])(x)
        x = Concatenate(axis=1)([con2_split1, con2_split2])
        x = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), name="conv2", activation="relu")(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="pool2")(x)
        x = BatchNormalization()(x)

        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), name="conv3", activation="relu")(x)
        
        x = ZeroPadding2D((1, 1))(x)
        con4_split1 = Lambda(lambda z: z[:,:,:,:192])(x)
        con4_split2 = Lambda(lambda z: z[:,:,:,192:])(x)
        x = Concatenate(axis=1)([con4_split1, con4_split2])
        x = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), name="conv4", activation="relu")(x)

        x = ZeroPadding2D((1, 1))(x)
        con5_split1 = Lambda(lambda z: z[:,:,:,:192])(x)
        con5_split2 = Lambda(lambda z: z[:,:,:,192:])(x)
        x = Concatenate(axis=1)([con5_split1, con5_split2])
        x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), name="conv5", activation="relu")(x)
        
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="pool5")(x)
        x = Flatten()(x)
        
        x = Dense(2048, activation='relu', name="fc6")(x)
        x = Dropout(0.5, name="droupout6")(x)
        x = Dense(2048, activation='relu', name="fc7")(x)
        x = Dropout(0.5, name="droupout7")(x)
        x = Dense(n_outputs, activation='softmax', name="fc8")(x)

        base_model = Model(inputs=model_input, outputs=x)
        base_model.load_weights("alexnet_weights.h5", by_name=True)
    
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
    # Available models: ['AlexNet_simplified', 'AlexNet_full', 'LeNet5', 'InceptionV3', 'MobileNetV2', 'ResNet50']
    
    avail_model = ['AlexNet_simplified', 'AlexNet_full', 'LeNet5', 'InceptionV3', 'MobileNetV2', 'ResNet50']
    
    # Getting data ready    
    df = pd.read_csv('out.csv', sep=',', header=None)
    
    X, y_temp = [], []
    
    # Ignore the df header
    for i in range(1, len(df.iloc[:,0])):
        X.append(cv2.imread(os.path.join(data_dir, df.iloc[:,1][i])))
        y_temp.append(ast.literal_eval(df.iloc[:,2][i]))
    
    # Prepare correct format for the input features X 
    X = np.array(X)

    # Fewer dataset
    X = X[:400]
    
    # Prepare correct format for the output label y (as one-hot vector)
    y_temp = np.array(y_temp)
    y = np.zeros((y_temp.size, y_temp.max() + 1))
    y[np.arange(y_temp.size), y_temp] = 1
    
    # Model selection with an iterator
    it = 0
    model, saved_name = create_model(avail_model[it])

    
    opt = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    # Training
    history = model.fit(X, y, epochs=50)
    
    '''
    # Saving trained model 
    h5_file_name = saved_name + '.h5'
    model.save(h5_file_name)
    '''
       
    # Load the saved model - use ONLY when inference (not to re-train again/call specific saved model)
    # model = tf.keras.models.load_model("traffic_status_AlexNet.h5")