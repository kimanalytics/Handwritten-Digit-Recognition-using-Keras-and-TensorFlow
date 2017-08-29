# Handwritten Digit Recognition using Keras and TensorFlow
## Introduction
In this project, I will develop a deep learning model to achieve a near state-of-the-art performance on the MNIST handwritten dataset. I'm going to use Keras with TensorFlow.

### MNIST Dataset
This dataset was constructed from a number of scanned document datasets availabe from the National Institute of Standards and Technology (NIST). These images were normalized in size and centered. Each image is in a 28x28 square (784 pixels). 60,000 images were used to train a model and 10,000 were used to test it. Excellent results achieve a prediction error of 1%. State-of-the-art results are approximately 0.2% which could be achieved with a large convolutional neural network.

## Baseline Model with Multilayer Perceptrons
We start with a baseline model so we can compare our convolutional neural network that we will use later.
To do a multilayer perceptron model, we flatten our 28 by 28 pixel images into a single 784 length vector for each image.
We then change the grayscale values from 0-255 to 0-1 to make things easier on our neural network. (Normalization)
Finally, we change the categories 1-9 into a binary matrix.
Our current neural network structure is as follows:

**Visible Layer (784 Inputs) >> Hidden Layer (784 Neurons) >> Output Layer (10 Outputs)**

## Simple Convolutional Neural Network for MNIST
As expected, we achieved around 1-2% error which is great. However, we can do better. Here, we take advantage of Kera's capability of creating convolutional neural networks. We will use all aspects of a modern CNN implementation, including convolutional layers, pooling layers, and dropout layers.
Here are our changes for the baseline model:
We add a convolutional layer with 32 feature maps, with a size of 5 x 5. This is also our input layer which expects images to be added.
We then define a pool size of 2 x 2.
We randomly dropout 20% of our neurons to reduce the amount of overfitting.
We then flatten our data.
We add 128 neurons with a rectifer activation function like above.
Finally we use 10 neurons for the 10 prediction classes with a softmax activation function to output probability-like prediction for each class.
Our current neural network structure is as follows:

**Visible Layer (1x28x28 Inputs) >> Convolutional Layer (32 maps, 5x5) >> Max Pooling Layer (2x2) >> Dropout Layer (20%) >> Flatten Layer >> Hidden Layer (128 Neurons) >> Output Layer (10 Outputs)**

## Larger Convolutional Neural Network for MNIST
Here we achieved around 1% error which is excellent. However, we can hit state-of-the-art results. Here, we deepen and widen our neural network.
Our current neural network structure is as follows:

**Visible Layer (1x28x28 Inputs) >> Convolutional Layer (30 maps, 5x5) >> Max Pooling Layer (2x2) >> Convolutional Layer (15 maps, 3x3) >> Max Pooling Layer (2x2) >> Dropout Layer (20%) >> Hidden Layer (128 Neurons) >> Hidden Layer (50 Neurons) >> Output Layer (10 Outputs)**

## Conclusion
With our ability to take advantage of larger convolutional neural network with Keras, we were able to go from 1-2% prediction error to less than 1%, near-state-of-the-art results! However, even with this model there are still further improvements which we can do with image augmentation and a much more powerful GPU.
