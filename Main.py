import os
import cv2
import numpy as np

try:
    # pydot-ng is a fork of pydot that is better maintained.
    import pydot_ng as pydot
except ImportError:
    # pydotplus is an improved version of pydot
    try:
        import pydotplus as pydot
    except ImportError:
        # Fall back on pydot if necessary.
        try:
            import pydot
        except ImportError:
            pydot = None

def _check_pydot():
    try:
        # Attempt to create an image of a blank graph
        # to check the pydot/graphviz installation.
        pydot.Dot.create(pydot.Dot())
    except Exception:
        # pydot raises a generic Exception here,
        # so no specific class can be caught.
        raise ImportError('Failed to import pydot. You must install pydot'
                          ' and graphviz for `pydotprint` to work.')

from ann_visualizer.visualize import ann_viz

from keras import backend as k
k.set_image_dim_ordering('tf')

from keras.utils import np_utils, plot_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import load_model
from keras.models import model_from_json

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split


dataPath = 'Fish Dataset\\Fish Dataset'
testDataPath = 'Fish Dataset\\Test'
numCategories = 10
dataDirList = os.listdir(dataPath)
testDataDirList = os.listdir(testDataPath)
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
sampleSize = imgData.shape[0]
labels = np.ones((sampleSize,), dtype='int64')

# Assign a label to each image so that they can be randomly shuffled whilst maintaining their
# category identifier
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

print('Loading Model...')

model = load_model('model\\modelNew.h5')

# inputShape = imgData[0].shape
#
# model = Sequential()
#
# model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=inputShape))
# model.add(Activation('relu'))
# model.add(Convolution2D(32, 3, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))
#
# model.add(Convolution2D(64, 3, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))
#
# model.add(Flatten())
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(numCategories))
# model.add(Activation('softmax'))
#
# model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=["accuracy"])
#
# model.summary()
# model.get_config()
# model.layers[0].get_config()
# # model.layers[0].input_shape
# # model.layers[0].output_shape
# model.layers[0].get_weights()
# np.shape(model.layers[0].get_weights()[0])
# # model.layers[0].trainable()
#
# epochNum = 30
#
# hist = model.fit(trainX, trainY, batch_size=16, nb_epoch=epochNum, verbose=1, validation_data=(testX, testY))
#
# model.save("model\\modelNew.h5")

flag = True
while flag:
    usrVis = input("Would you like to visualise the model? Y/N: ")
    if usrVis == 'Y':
        try:
            ann_viz(model, 'visualisation.gv', title="Convolutional Neural Network for Fish Classification")
        except OSError:
            print(OSError)
        flag = False
    elif usrVis == 'N':
        flag = False
    else:
        print("Invalid Input")


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
    usrTst = input("Would you like to test using images outside the dataset? Y/N: ")
    if usrTst == 'Y':
        print('\n')
        try:
            imgTestList = os.listdir(testDataPath)
            for img in imgTestList:
                inputTestImg = cv2.imread(testDataPath + '/' + img)
                testImage = cv2.resize(inputTestImg, (imgRows, imgCols))
                imgDataList.append(testImage)

                testImage = cv2.resize(testImage, (imgRows, imgCols))
                testImage = np.array(testImage)
                testImage = testImage.astype('float32')
                testImage /= 255
                testImage = np.expand_dims(testImage, axis=0)

                predict = (model.predict_classes(testImage))

                predict = predict.astype('int')
                prediction = predict[0]
                print('The most likely category for: ', img, ' is ', categories[prediction], '\n')
            flag = False
        except OSError:
            print("Invalid File Name")
    elif usrTst == 'N':
        flag = False
    else:
        print('Invalid Input')
