#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 13:55:45 2020

@author: gama
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow import keras
import cv2


from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions


"""
modelName is a string; example : vgg16
"""

#LOAD MODEL.


vgg16_model = tf.keras.applications.vgg16.VGG16()

vgg16_model.summary()

#%%
model= keras.Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)
    
    
model.summary()
    
    
#%% Make the layers not trainable
    
for layer in model.layers:
    layer.trainable = False
    

model.summary()


#%% Add last dense layer for the 10 classes

model.add(keras.layers.Dense(units=10,activation = 'softmax'))

model.summary()


#%%


mnist = keras.datasets.mnist
num_classes=10
input_shape = (28, 28, 1)
#Split data into training sets & test sets
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#Scale all values from [0;255] to [0;1] 

#train_images = train_images / 255.0
#test_images = test_images / 255.0
# Scale images to the [0, 1] range
train_images = train_images.astype("float32") / 255
test_images = test_images.astype("float32") / 255

# Make sure images have shape (28, 28, 1)
train_images = np.expand_dims(train_images, -1)
test_images = np.expand_dims(test_images, -1)
    
    
# convert class vectors to binary class matrices
train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels =  keras.utils.to_categorical(test_labels, num_classes)


#train_images= cv2.reshape()
#%%
print(train_images.shape)
train_images = cv2.cvtColor(train_images,cv2.COLOR_GRAY2RGB)
#train_images = cv2.resize(224,224)

print(train_images.shape)


#%%

model.summary()
model.compile(optimizer='adam',
          loss='categorical_crossentropy',
          metrics=['accuracy'])


model.fit(train_images, train_labels, epochs=10)



















#loadKerasModel("vgg16")
