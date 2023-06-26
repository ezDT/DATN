import ast
import cv2
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re
import sys
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import time

# %% Main function

if __name__ == "__main__":
    # Choose a model's name
    # Available models: ['AlexNet', 'LeNet5', 'InceptionV3', 'MobileNetV2', 'ResNet50']
    
    avail_model = ['AlexNet_simplified', 'AlexNet_full', 'LeNet5', 'InceptionV3', 'MobileNetV2', 'ResNet50', 'VGG19']
    
    # Getting data ready
    training_image_dir = 'F:/dataset/all_resized' 
    
    df = pd.read_csv('out_fewer.csv', sep=',', header=None)
    
    X, y_test = [], []
    
    # Ignore the df header
    for i in range(1, len(df.iloc[:,0])):
        X.append(cv2.imread(os.path.join(training_image_dir, df.iloc[:,1][i])))
        y_test.append(ast.literal_eval(df.iloc[:,2][i]))
    
    # Prepare correct format for the input features X 
    X = np.array(X)
    
    # Prepare correct format for the output label y (as one-hot vector)
    y_test = np.array(y_test)
    y = np.zeros((y_test.size, y_test.max() + 1))
    y[np.arange(y_test.size), y_test] = 1
    
    y_test = y
    
    for i in range(2,3):
        # Model selection with an iterator
        it = i
        
        # Load the saved model - use ONLY when inference (not to re-train again/call specific saved model)
        if it == 0:
            model = tf.keras.models.load_model("traffic_status_mini_AlexNet.h5")
        elif it == 1:
            model = tf.keras.models.load_model("traffic_status_original_AlexNet.h5")
        elif it == 2:
            model = tf.keras.models.load_model("traffic_status_modified_LeNet5.h5")
        elif it == 3:
            model = tf.keras.models.load_model("traffic_status_InceptionV3.h5")
        elif it == 4:
            model = tf.keras.models.load_model("traffic_status_MobileNetV2.h5")
        elif it == 5:
            model = tf.keras.models.load_model("traffic_status_ResNet50.h5")
        elif it == 6:
            model = tf.keras.models.load_model("traffic_status_VGG19.h5")
                
        # Predict for 2D input image
        start_time = time.time()
        y_pred = model.predict(X, batch_size=2)
        end_time = time.time()
        
        # One-hot decode for y_pred
        one_hot_decode = to_categorical(np.argmax(y_pred, axis=1), 5)
        
        print(f"Traffic status inference time: {(end_time-start_time):.3f}s")
        
        print("Model: ", avail_model[it])
        print("Accuracy:", accuracy_score(y_test, one_hot_decode))
        print("Precision:", precision_score(y_test, one_hot_decode, average="weighted"))
        print('Recall:', recall_score(y_test, one_hot_decode, average="weighted"))
        print('F1 score:', f1_score(y_test, one_hot_decode, average="weighted"))
    
    cv2.waitKey(0)