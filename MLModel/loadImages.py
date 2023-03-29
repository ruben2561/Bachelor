import glob
import cv2
import os
import numpy as np

def load_images(directory):
    images = []
    labels = []

    files = glob.glob('C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/text_images/*')

    for filename in files:
        # Extract the word from the image name
        word = filename.split('__')[0]
        
        img = cv2.imread(os.path.join(directory, filename))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #img = cv2.resize(img, (50, 50))  # Resize the image to a smaller size
            images.append(img)
            labels.append(word)
    return np.array(images), np.array(labels)