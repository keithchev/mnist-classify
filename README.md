# mnist-classify

Using a CNN with random affine transformations to classify MNIST digits with an error rate of ~.31%

Notes/remarks:

The CNN architecture is a modified version of that found in the mnist_cnn example in the Keras doc.

The training data is subject to random affine distortions at each epoch:
    - random shift in x-y by [-3,3] pixels
    - random rotation by [-10, 10] degrees
    - random resize by [.8, 1.2] factor
  These parameters were determined empirically by inspecting the original training data.

At each epoch, the model is trained only on the distorted data, and validated on the entire original 
training data. 

The model was trained > 60 times with an increasing batch size (100 to 1000), predictions saved 
after each epoch, and a final prediction generated from the consensus of all predictions 
with a validation accuracy greater than .999. 

Ran on an EC2 g2.2xlarge instance (GRID K520 GPU) with CUDA 7.5, Tensorflow 0.9 and Keras 1.1.1 
(nb these are all old versions) 
