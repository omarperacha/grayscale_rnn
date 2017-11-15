'''
Created on 15 Nov 2017

@author: omarperacha
'''

import numpy as np
from PIL import Image
import glob
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import RMSprop

#CHANGE PATH AND FILETYPE IN glob.glob TO MATCH YOUR TRAINING DATA
images=glob.glob("/Users/NAME/training_set/*.jpeg")
array_length = len(images)

#CHANGE TO True IF USING CROSS VALIDATION
cross_validating = False

if cross_validating:
    #CHANGE PATH AND FILETYPE IN glob.glob TO MATCH YOUR CROSS VALIDATION DATA
    cv_images=glob.glob("/Users/NAME/cross_validation_set/*.jpeg")
    cv_array_length = len(cv_images)

#CHANGE THESE VALUES TO MATCH YOUR TRAINING IMAGES
image_width = 60
image_height = 90

n_pixels = image_width*image_height

#CHANGE seq_length TO ANOTHER VALUE IF DESIRED
seq_length = image_width
dataX = []
dataY = []
#convert each image's pixel values into a column vector
for i in range(array_length):
    img = Image.open(images[i])
    data = np.array(img)
    data = np.reshape(data, n_pixels)
    #prepare X & y data
    for i in range(0, n_pixels - seq_length, 1):
        seq_in = data[i:i + seq_length]
        seq_out = data[i + seq_length]
        dataX.append(seq_in)
        dataY.append(seq_out)
n_patterns = len(dataX)
print ("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(255)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

if cross_validating:
    cvX = []
    cvY = []
    for i in range(cv_array_length):
        cv_img = Image.open(cv_images[i])
        cv_data = np.array(img)
        cv_data = np.reshape(data, n_pixels)
        for i in range(0, n_pixels - seq_length, 1):
            cv_seq_in = cv_data[i:i + seq_length]
            cv_seq_out = cv_data[i + seq_length]
            cvX.append(cv_seq_in)
            cvY.append(cv_seq_out)
    n_cv_patterns = len(cvX)
    print ("Total cv_Patterns: ", n_cv_patterns)
    cvX = np.reshape(cvX, (n_cv_patterns, seq_length, 1))
    cvX = cvX / float(255)
    cvY = np_utils.to_categorical(cvY)
    
# define the LSTM model
model = Sequential()
model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

#UNCOMMENT NEXT TWO LINES TO LOAD YUOR OWN WEIGHTS (OR THE ONES PROVIDED)
#filename = "weights-improvement-00-0.4125-3.hdf"
#model.load_weights(filename)

#CHANGE lr TO ADJUST LEARNING RATE AS YOU DESIRE. (A DECAYING RATE WORKS WELL).
rms = RMSprop(lr=0.01)

if cross_validating:
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
else: 
    model.compile(loss='categorical_crossentropy', optimizer=rms)
    
# checkpoint after each training epoch - weights saved only if loss improves
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# fit the model
if cross_validating:
    model.fit(X, y, validation_data=(cvX, cvY), epochs=2000, batch_size=450, callbacks=callbacks_list)
else:
    model.fit(X, y, epochs=2000, batch_size=450, callbacks=callbacks_list)