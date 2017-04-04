'''
Using a CNN with random affine transformations to classify MNIST digits 

Keith Cheveralls
March 2017

Notes:

    The CNN architecture is a modified version of that found in the mnist_cnn example in the Keras documentation.

    The training data is subject to random affine distortions at each epoch:
        1) random shift in x-y by [-3,3] pixels
        2) random rotation by [-10, 10] degrees
        3) random resize by [.8, 1.2] factor

    These parameters were determined empirically by inspecting the original training images.

    At each epoch, the model is trained only on the distorted data, and validated on the entire original 
    training data. 

    The model was trained > 60 times with an increasing batch size (100 to 1000), predictions saved 
    after each epoch, and a final prediction generated from the consensus of all predictions 
    with a validation accuracy greater than .999. 

    Ran in a few hours on an EC2 g2.2xlarge instance (GRID K520 GPU) with CUDA 7.5, Tensorflow 0.9 and Keras 1.1.1 
    (nb these are all old versions, an runtime was limited by affine transformations running on CPU) 

Example usage:

    # load training images
    x, y = loadKaggle('./data/train.h5')

    # load test images
    xtest = loadKaggleTest('./data/test.h5')

    # initialize Keras model
    model = loadModel()

    # initial training on original data only to get a reasonably accurate net 
    # no need to worry about overfitting because later epochs on distorted data
    runModel(x, y, x, y, model, [], 250, 30, False)

    # refinement using distorted data (w/ prediction saved at each epoch)
    # val accuracy seems to fluctuate around .990-.998 - probably more room for improvement
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
    ''' Load training images from Kaggle CSV, re-saved as HDF5
    This is an Nx28x28 stack of digit images (N = 42000) '''
    
    data = pd.read_hdf(fname, 'table')
    data = np.array(data).astype('float32')

    x = data[:, 1:]
    y = data[:, 0]
    
    x /= 255
    
    # the image shape needs to be (28, 28, 1) (and not (1, 28, 28))
    x = x.reshape(x.shape[0], 28, 28, 1)

    return x, y


def loadKaggleTest(fname):
    ''' Load test data for Kaggle '''
    
    data = pd.read_hdf(fname, 'table')  

    data = np.array(data).astype('float32')/255
    data = data.reshape(data.shape[0], 28, 28, 1)

    return data


def loadModel():
    ''' Load the CNN we're going to use - inspired by Keras' mnist_cnn example'''
    
    # number of output classes 
    num_classes = 10
    
    # hard-coded input shape
    input_shape = (28, 28, 1)

    # ------------------------
    # Building the CNN 
    # ------------------------
    
    # 3x3 kernels seem to work better here than 5x5 (and are faster, obviously)
    filter_size = (3, 3)
    
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=input_shape))
    
    model.add(Convolution2D(32, 3, 3, activation='relu')) # omitting pooling before this layer seems to work well (and is faster)
    model.add(MaxPooling2D(pool_size=(2, 2))) 

    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  
    model.add(Dense(1024, activation='relu')) 
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # (adam is faster than adadelta here)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    return model


def runModel(x, y, x_validate, y_validate, model, x_predict=[], batch_size=128, epochs=1, distortFlag=False):
    ''' Train the CNN (with or without affine transformations) '''
    
    num_classes = 10

    # convert class vectors to binary class matrices
    y = to_categorical(y, num_classes)
    y_validate  = to_categorical(y_validate, num_classes)

    # randomly affine-distort the training set
    if distortFlag:
        x = randomAffine(x)

    model.fit(x, y, 
              batch_size=batch_size, 
              nb_epoch=epochs, 
              show_accuracy=True,
              validation_data=(x_validate, y_validate), 
              shuffle=True)
    
    score = model.evaluate(x_validate, y_validate, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    if len(x_predict):
        yp = makePredictions(x_predict, model)
        savePredictions(yp, './data/mnist_v3_acc%s.csv' % score[1])

    return model


def makePredictions(x, model):
    ''' Predict classes given images and model. '''
    
    yp = model.predict(x)

    # class IDs from softmax probabilities
    yp = yp.argmax(axis=1)

    return yp


def savePredictions(yp, fname):
    ''' Save predictions in appropriate format to upload to kaggle '''
    
    ytestd = np.array([np.arange(1,len(yp)+1), yp]).transpose()
    ytestd = pd.DataFrame(ytestd, columns=["ImageID", "Label"])
    ytestd.to_csv(fname, index=False)


def randomAffine(x):
    ''' Apply a random translation+rotation+scale to each image in x.
    This is used to generate 'new' training data at each epoch '''
    
    # placeholder array for distorted images
    x_ = 0*x

    shiftRange = [-3, 3]
    thetaRange = [-10, 10]
    scaleRange = [.8, 1.2]

    reshapeFlag = len(x.shape)==2
    
    for ind in np.arange(0, x.shape[0]):

        # pick a random shift, rotation, and scale
        shift = np.random.rand(2) * (np.max(shiftRange) - np.min(shiftRange)) + shiftRange[0]
        theta = np.random.rand() * (np.max(thetaRange) - np.min(thetaRange)) + thetaRange[0]
        scale = np.random.rand() * (np.max(scaleRange) - np.min(scaleRange)) + scaleRange[0]

        if reshapeFlag:
            im = x[ind, :].reshape((28,28))
        else:
            im = x[ind, :, :, 0]

        # ----------------------
        # distort the image
        #-----------------------
        
        # note: probably important to use interp='nearest' and order=0 in the functiona below,
        # in order to best preserve the sharp edges of the original images
        
        # rotation
        im = sp.misc.imrotate(im, theta, interp='nearest')

        # scale (note: the offset required to keep image centered)
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





