'''
Uses a CNN with random affine transformations to classify MNIST digits 

The CNN architecture is a modified version of that found in the mnist_cnn example in the Keras doc.

The training data is subject to random affine distortions at each epoch:
    1) random shift in x-y by [-3,3] pixels
    2) random rotation by [-10, 10] degrees
    3) random resize by [.8, 1.2] factor

These parameters were determined empirically by inspecting the original training data.

At each epoch, the model is trained only on the distorted data, and validated on the entire original 
training data. 

The model was trained > 60 times with an increasing batch size (100 to 1000), predictions saved 
after each epoch, and a final prediction generated from the consensus of all predictions 
with a validation accuracy greater than .999. 

Ran on an EC2 g2.2xlarge instance (GRID K520 GPU) with CUDA 7.5, Tensorflow 0.9 and Keras 1.1.1 
(nb these are all old versions) 

Example usage:

  x, y = loadKaggle('./data/train.h5')
  xtest = loadKaggleTest('./data/test.h5')

  model = loadModel()

  # initial training on original data to get a reasonably accurate net
  runModel(x, y, x, y, model, [], 250, 30, False)

  # refinement using distorted data (prediction saved at each epoch)
  runModel(x, y, x, y, model, xtest, 500, 30, True)



'''

import gc
import numpy as np
import pandas as pd

import scipy as sp
import scipy.misc as spmisc
from scipy import ndimage

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D

def loadKaggle(fname):

    # load data from Kaggle CSV (saved as .h5)
    # note that input_shape is assumed to be (nrows, ncols, 1)

    data = pd.read_hdf(fname, 'table')
    data = np.array(data).astype('float32')

    x = data[:, 1:]
    y = data[:, 0]

    del data
    gc.collect()

    x /= 255
    x = x.reshape(x.shape[0], 28, 28, 1)

    return x, y


def loadKaggleTest(fname):

    # load test data for Kaggle - assume h5

    data = pd.read_hdf(fname, 'table')  

    data = np.array(data).astype('float32')/255
    data = data.reshape(data.shape[0], 28, 28, 1)

    return data


def loadModel():

    num_classes = 10
    input_shape = (28, 28, 1)

    # CNN architecture for classification
    # note: 3x3 kernels seem to work better here than 5x5 (and are faster, obviously)
    
    filter_size = (3, 3)
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=input_shape))

    # this second conv layer without pooling seems to work well (and is faster)
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) 

    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  
    model.add(Dense(1024, activation='relu')) 
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # adam is faster than adadelta
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    return model


def runModel(x, y, x_validate, y_validate, model, x_predict=[], batch_size=128, epochs=1, distortFlag=False):

    num_classes = 10

    # convert class vectors to binary class matrices
    y = to_categorical(y, num_classes)
    y_validate  = to_categorical(y_validate, num_classes)

    # randomly distort the training set
    if distortFlag:
        x = randomAffine(x)

    model.fit(x, y, batch_size=batch_size, nb_epoch=epochs, show_accuracy=True,
      validation_data=(x_validate, y_validate), shuffle=True)
    
    score = model.evaluate(x_validate, y_validate, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    if len(x_predict):
        yp = makePredictions(x_predict, model)
        savePredictions(yp, './data/mnist_v3_acc%s.csv' % score[1])

    return model

def makePredictions(x, model):

    yp = model.predict(x)

    # class IDs from softmax probabilities
    yp = yp.argmax(axis=1)

    return yp

def savePredictions(yp, fname):

    #  save predictions in appropriate format to upload to kaggle

    ytestd = np.array([np.arange(1,len(yp)+1), yp]).transpose()

    ytestd = pd.DataFrame(ytestd, columns=["ImageID", "Label"])

    ytestd.to_csv(fname, index=False)


def randomAffine(x):

    # apply a random translation+rotation+scale to each image in x
    # used to generate 'new' training data at each epoch 
    
    x_ = 0*x
    
    N = x.shape[0]

    shiftRange = [-3, 3]
    thetaRange = [-10, 10]
    scaleRange = [.8, 1.2]

    reshapeFlag = len(x.shape)==2
    
    for ind in np.arange(0, N):

        shift = np.random.rand(2) * (np.max(shiftRange) - np.min(shiftRange)) + shiftRange[0]
        theta = np.random.rand() * (np.max(thetaRange) - np.min(thetaRange)) + thetaRange[0]
        scale = np.random.rand() * (np.max(scaleRange) - np.min(scaleRange)) + scaleRange[0]

        if reshapeFlag:
            im = x[ind, :].reshape((28,28))
        else:
            im = x[ind, :, :, 0]

        # probably important to use nn interp here (and order=0 spline interp below)
        # in order to best preserve the sharp edges of the original images
        
        # rotation
        im = sp.misc.imrotate(im, theta, interp='nearest')

        # scale (note: offset required to keep image centered)
        im = ndimage.affine_transform(im, np.eye(2)*scale, offset=np.array([14,14])*(1-scale), order=0)

        # translation 
        im = ndimage.shift(im, (shift[0], shift[1]), order=0)
        
        if reshapeFlag:
            x_[ind, :] = im.flatten()
        else:
            x_[ind, :, :, 0] = im

    if x.dtype=='uint8':
        return x_
    else:
        return x_/255




# --- copied from current keras 2.0 source --- #
def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical





