import ast
import cv2
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import sys
import tensorflow as tf
import time

# %% Main function

if __name__ == "__main__":
    # Choose a model's name
    # Available models: ['AlexNet', 'LeNet5', 'InceptionV3', 'MobileNetV2', 'ResNet50']
    
    avail_model = ['AlexNet', 'LeNet5', 'InceptionV3', 'MobileNetV2', 'ResNet50']
    
    # Hide GPU from visible devices
    tf.config.set_visible_devices([], 'GPU')

    result = []

    image = cv2.imread('F:/dataset/all_resized/20230402_144533_000022.jpg')
    
    image = cv2.resize(image, (640, 360), interpolation = cv2.INTER_AREA)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    # Load the saved model - use ONLY when inference (not to re-train again/call specific saved model)
    model = tf.keras.models.load_model("traffic_status_VGG19.h5")
    
    # Predict for 2D input image
    X_test = []
    X_test.append(image_rgb)
    X_test = np.array(X_test)
    
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()

    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()
    
    print(y_pred[0])
    
    print(f"Traffic status inference time: {(end_time-start_time):.3f}s")
    
    cv2.waitKey(0)