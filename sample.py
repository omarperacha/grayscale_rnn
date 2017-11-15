'''
Created on 15 Nov 2017

@author: omarperacha
'''

import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils


#CHANGE PATH AND FILETYPE IN glob.glob TO MATCH YOUR TRAINING DATA
images=glob.glob("/Users/NAME/training_set/*.jpeg")
array_length = len(images)

#CHANGE THESE VALUES TO MATCH YOUR TRAINING IMAGES
image_width = 60
image_height = 90

n_pixels = image_width*image_height

# prepare seeds to start image generation with
seq_length = image_width
dataX = []
dataY = []
for i in range(array_length):
    img = Image.open(images[i])
    data = np.array(img)
    data = np.reshape(data, n_pixels)
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
# define the LSTM model
model = Sequential()
model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

#CHANGE filename TO LOAD YOUR OWN WEIGHTS
filename = "weights-improvement-00-0.4125-3.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# pick a random seed
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
pattern_temp = pattern
print ("starting")

# generate image
for i in range(n_pixels-seq_length):
    x = np.reshape(pattern_temp, (1, len(pattern_temp), 1))
    print(i)
    x = x / float(255)
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    print(index)
    pattern = np.append(pattern, index)
    pattern_temp = pattern[(1+i):len(pattern)]
print ("Done.")
pattern = np.reshape(pattern,(image_height, image_width))
plt.imshow(pattern, cmap='Greys_r', interpolation='nearest')
plt.show()