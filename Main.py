import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras import backend as k
k.set_image_dim_ordering('tf')

from keras.utils import np_utils, plot_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import  Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split


dataPath = 'Fish Dataset\\Fish Dataset'
numCategories = 10
dataDirList = os.listdir(dataPath)
imgRows = 128
imgCols = 128
numChannels = 3
imgDataList = []

# load each dataset category and resize each image for processing
for dataset in dataDirList:
    imgList = os.listdir(dataPath + '/' + dataset)
    print('\n' + '{}'.format(dataset) + ' Dataset Loaded, Resizing Images...')
    for img in imgList:
        inputImg = cv2.imread(dataPath + '/' + dataset + '/' + img)
        inputImgResize = cv2.resize(inputImg, (imgRows, imgCols))
        imgDataList.append(inputImgResize)
    print('Resizing complete.  moving to next category...')

print('Finished Resizing\n')
imgData = np.array(imgDataList)
imgData = imgData.astype('float32')
imgData /= 255
# print('(Samples, Rows, Cols, Dimensions) = ')
# print(imgData.shape)

sampleSize = imgData.shape[0]
labels = np.ones((sampleSize,), dtype='int64')

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

x, y = shuffle(imgData, lc, random_state=2)

trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.2, random_state=2)

print('Beginning Training Using Keras and TensorFlow')

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
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
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

epochNum = 15

hist = model.fit(trainX, trainY, batch_size=16, nb_epoch=epochNum, verbose=1, validation_data=(testX, testY))


flag = True
while flag:
    usrPlt = input("Would you like to plot Training and Validation Data? Y/N: ")
    if usrPlt == 'Y':
        # plot_model(model, to_file='model.png', show_shapes=False, show_layer_names=True)
        # visualizing losses and accuracy
        trainLoss = hist.history['loss']
        valLoss = hist.history['val_loss']
        trainAcc = hist.history['acc']
        valAcc = hist.history['val_acc']
        e = range(epochNum)
        plt.figure(1, figsize=(7, 5))
        plt.plot(e, trainLoss)
        plt.plot(e, valLoss)
        plt.xlabel('num of Epochs')
        plt.ylabel('loss')
        plt.title('training loss vs validation loss')
        plt.grid(True)
        plt.legend(['train', 'val'])
        # print plt.style.available # use bmh, classic,ggplot for big pictures
        plt.style.use(['classic'])
        plt.show()

        plt.figure(2,figsize=(7,5))
        plt.plot(e, trainAcc)
        plt.plot(e, valAcc)
        plt.xlabel('num of Epochs')
        plt.ylabel('accuracy')
        plt.title('train accuracy vs validation accuracy')
        plt.grid(True)
        plt.legend(['train', 'val'], loc=4)
        # print plt.style.available # use bmh, classic,ggplot for big pictures
        # plt.style.use(['classic'])
        plt.show()
        flag = False
    elif usrPlt == 'N':
        flag = False
    else:
        print('Invalid Input')

flag = True
while flag:
    usrEval = input("Would you like to evaluate this model? Y/N: ")
    if usrEval == 'Y':
        score = model.evaluate(testX, testY, batch_size=128)
        print('Test Loss', score[0])
        print('Test Accuracy', score[1])
        flag = False
    elif usrEval == 'N':
        flag = False
    else:
        print('Invalid Input')

flag = True
while flag:
    usrTst = input("Would you like to test using an image outside the dataset? Y/N: ")
    if usrTst == 'Y':
        try:
            fishTest = input("Place the image to be tested in the \"Fish Dataset\\Test\\\" Folder and provide the file"
                             " name and extension now (only jpg files supported): ")
            testImage = cv2.imread('Fish Dataset\\Test\\' + fishTest + '.jpg')
            testImage = cv2.resize(testImage, (imgRows, imgCols))
            testImage = np.array(testImage)
            testImage = testImage.astype('float32')
            testImage /= 255
            testImage = np.expand_dims(testImage, axis=0)
            print(testImage.shape)

            print((model.predict(testImage)))
            predict = (model.predict_classes(testImage))

            predict = predict.astype('int')
            prediction = predict[0]
            print('The most likely category is : ' + categories[prediction])
            flag = True
        except OSError:
            print("Invalid File Name")
    elif usrTst == 'N':
        flag = False
    else:
        print('Invalid Input')
