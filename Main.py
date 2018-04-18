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

i = 0
#load each dataset category and resize each image for processing
for dataset in dataDirList:
    imgList = os.listdir(dataPath + '/' + dataset)
    print ('\n' + '{}'.format(dataset) + ' Dataset Loaded, Resizing Images...')
    for img in imgList:
        i = i + 1
        inputImg = cv2.imread(dataPath + '/' + dataset + '/' + img)
        inputImgResize = cv2.resize(inputImg, (imgRows, imgCols))
        imgDataList.append(inputImgResize)
    print(i)
    print('Resizing complete.  moving to next dataset...')

print('Finished Resizing\n')
imgData = np.array(imgDataList)
imgData = imgData.astype('float32')
imgData /= 255
print('(Samples, Rows, Cols, Dimensions) = ')
print(imgData.shape)



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

humanLabels = ['bluestripeSnapper', 'cardinalfish', 'clownfish', 'clownTriggerfish',
               'lionfish', 'moorishIdol', 'parrotfish', 'saddleButterflyfish',
               'spottedTrunkfish', 'yellowTang']
labelsToHuman = np_utils.to_categorical(humanLabels, numCategories)



