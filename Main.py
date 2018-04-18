import os
import cv2
import numpy as np
import matplotlib.pyplot as plot

from keras import backend as k
import tensorflow as tf
k.set_image_dim_ordering('tf')

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import  Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split


dataPath = 'C:\\Users\\Thoma\\Fish Dataset\\Fish Dataset'
numCategories = 10
dataDirList = os.listdir(dataPath)
imgRows = 256
imgCols = 256
numChannels = 3

imgDataList = []

#load each dataset category and resize each image for processing
for dataset in dataDirList:
    imgList = os.listdir(dataPath + '/' + dataset)
    print ('\n' + '{}'.format(dataset) + ' Dataset Loaded, Resizing Images...')
    for img in imgList:
        inputImg = cv2.imread(dataPath + '/' + dataset + '/' + img)
        inputImgResize = cv2.resize(inputImg, (imgRows, imgCols))
        imgDataList.append(inputImgResize)

    print('Resizing complete.  moving to next category...')

print('Finished Resizing\n')
imgData = np.array(imgDataList)
imgData = imgData.astype('float32')
imgData /= 255
#print('(Samples, Rows, Cols, Dimensions) = ')
#print(imgData.shape)



sampleSize = imgData.shape[0]
labels = np.ones((sampleSize,),dtype='int64')

labels[0:150] = 0
labels[150:539] = 1
labels[539:695] = 2
labels[695:899] = 3
labels[899:1083] = 4
labels[1083:1235] = 5
labels[1235:1523] = 6
labels[1523:1616] = 7
labels[1616:1770] = 8
labels[1770:1810] = 9

categories = ['bluestripeSnapper', 'cardinalfish', 'clownfish', 'clownTriggerfish',
               'lionfish', 'moorishIdol', 'parrotfish', 'saddleButterflyfish',
               'spottedTrunkfish', 'yellowTang']
lc = np_utils.to_categorical(labels, numCategories)

x,y = shuffle(imgData, lc, random_state=2)

trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.2, random_state=2)


inputShape = imgData[0].shape

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=inputShape))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
# model.add(Convolution2D(64, 3, 3))
# model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(numCategories))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta',metrics=["accuracy"])

model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape
model.layers[0].output_shape
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable

training = model.fit(trainX, trainY, batch_size=16, nb_epoch=20, verbose=1, validation_data=(testX, testY))

testImage = cv2.imread('C:\\Users\\Thoma\\Fish Dataset\\Test\\test_fish.jpg')
testImage = cv2.resize(testImage, (imgRows, imgCols))
testImage = np.array(testImage)
testImage = testImage.astype('float32')
testImage /= 255
print (testImage.shape)

print((model.predict(testImage)))
print(model.predict_classes(testImage))