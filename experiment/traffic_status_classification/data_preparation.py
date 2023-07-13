import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# %% Preparation of the dataset

def get_data_shape():
    img_shape = (640, 360)
    return img_shape


def data_preparation():
    df = pd.DataFrame()
    
    file_list, label_list = [], []
    
    # Hard-coded data_dir path 
    
    data_dir = r'D:/Working/TLU/DATN/experiment/dataset/'
    original_dir = data_dir + 'original/'
    resized_dir = data_dir + 'resized/'
    
    #folder = ['rat_vang', 'vang', 'trung_binh', 'dong', 'tac']
    
    for folder in os.listdir(original_dir):
        for file in os.listdir(os.path.join(original_dir, folder)):
                        
            if file.endswith('.jpg'):
                img = cv2.imread(os.path.join(original_dir, folder, file))
                img = cv2.resize(img, get_data_shape())
                
                cv2.imwrite(os.path.join(resized_dir, folder, file), img)
                                
                file_list.append(file)
                
            if folder == '1 rat vang':
                label = 0
            elif folder == '2 vang':
                label = 1
            elif folder == '3 trung binh':
                label = 2
            elif folder == '4 dong':
                label = 3
            elif folder == '5 tac':
                label = 4
                
            label_list.append(label)
                
    df['filename'] = file_list
    df['label'] = label_list
            
    return df

# %% Main function

if __name__ == "__main__":
    df = data_preparation()
    df.to_csv('./out.csv')