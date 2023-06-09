# Importing modules 
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import cv2

from keras.utils import to_categorical
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout
from keras.models import Sequential

from sklearn.model_selection import train_test_split

np.random.seed(1)


def procces_training_data():
    # Processing training data
    # -> appending images in a list 'train_images'
    # -> appending labels in a list 'train_labels'

    train_images = []       
    train_labels = []
    shape = (200,200) #changed to 100,100 was 200,200 
    train_path = 'C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/dataset/nummers1/'

    for filename in os.listdir('C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/dataset/nummers1'):
        if filename.split('.')[1] == 'jpg':
            img = cv2.imread(os.path.join(train_path,filename))
            
            # Spliting file names and storing the labels for image in list
            train_labels.append(filename.split('_')[0])
            
            # Resize all images to a specific shape
            img = cv2.resize(img,shape)
            
            train_images.append(img)

    # Converting labels into One Hot encoded sparse matrix
    train_labels = pd.get_dummies(train_labels).values

    # Converting train_images to array
    train_images = np.array(train_images)

    # Splitting Training data into train and validation dataset
    x_train,x_val,y_train,y_val = train_test_split(train_images,train_labels,random_state=1)

    # Visualizing Training data
    print(train_labels[0])
    #plt.imshow(train_images[0])

    return x_train, x_val, y_train, y_val

def procces_testing_data():
    # Processing testing data
    # -> appending images in a list 'test_images'
    # -> appending labels in a list 'test_labels'
    # The test data contains labels as well also we are appending it to a list but we are'nt going to use it while training.

    test_images = []
    test_labels = []
    shape = (200,200)
    test_path = '../input/fruit-images-for-object-detection/test_zip/test'

    for filename in os.listdir('../input/fruit-images-for-object-detection/test_zip/test'):
        if filename.split('.')[1] == 'jpg':
            img = cv2.imread(os.path.join(test_path,filename))
            
            # Spliting file names and storing the labels for image in list
            test_labels.append(filename.split('_')[0])
            
            # Resize all images to a specific shape
            img = cv2.resize(img,shape)
            
            test_images.append(img)
            
    # Converting test_images to array
    test_images = np.array(test_images)

def create_model():
    # Creating a Sequential model
    model= Sequential()
    model.add(Conv2D(kernel_size=(3,3), filters=32, activation='tanh', input_shape=(200,200,3,)))
    model.add(Conv2D(filters=30,kernel_size = (3,3),activation='tanh'))
    model.add(MaxPool2D(2,2))
    model.add(Conv2D(filters=30,kernel_size = (3,3),activation='tanh'))
    model.add(MaxPool2D(2,2))
    model.add(Conv2D(filters=30,kernel_size = (3,3),activation='tanh'))

    model.add(Flatten())

    model.add(Dense(20,activation='relu'))
    model.add(Dense(15,activation='relu'))
    model.add(Dense(10,activation = 'softmax')) ################ changed from 4 -> 10
        
    model.compile(
                loss='categorical_crossentropy', 
                metrics=['acc'],
                optimizer='adam'
                )
    # Model Summary
    model.summary()

    return model

def predict():
    # Testing predictions and the actual label
    checkImage = test_images[0:1]
    checklabel = test_labels[0:1]

    predict = model.predict(np.array(checkImage))

    output = { 0:'apple',1:'banana',2:'mixed',3:'orange'}

    print("Actual :- ",checklabel)
    print("Predicted :- ",output[np.argmax(predict)])


################################################################
##########################"MAIN"################################
################################################################

# Training the model
x_train, x_val, y_train, y_val = procces_training_data()
model = create_model()

history = model.fit(x_train,y_train,epochs=50,batch_size=50,validation_data=(x_val,y_val))
model.save('C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/')

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()








################################################################
################################################################
################################################################