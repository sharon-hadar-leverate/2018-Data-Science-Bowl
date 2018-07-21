# Progress Report Introduction: 
This README file is a progress report for the project "2018-Data-Science-Bowl".    
In this project, I am going to explore the possibilities of deep learning and segmentation problem assuming that combining ladders networks will enhance the hidden network learning features.  
#### In this report i will introduce: 

### Part One:  :last_quarter_moon:
- [X] The project description 
- [X] A brif review of basic techniques in image segmentation
- [X] Deep Learning consepts
- [X] Convolutional Neural Network (CNN)
- [X] Use of Deep Learning in segmentation problem
- [ ] Description of primary products
### Part Two: :new_moon:
- [ ] Autoencoders
- [ ] Ladder Networks
- [ ] Use of Ladder Network in a CNN
- [ ] Use of Ladder Network in segmentation problem
- [ ] Description of primary products
- [ ] Bibliography

## The project description
_“2018 Data Science Bowl” is a Kaggle competition that its goal is to create an algorithm to automate nucleus detection in divergent images to advance medical discovery._  

_By observing patterns, asking questions, and building a model, participants will have a chance to push state-of-the-art technology farther._  

in this competition, the challanger expose a dataset contains a large number of segmented nuclei images.  
The images were acquired under a variety of conditions and vary in the cell type, magnification, and imaging modality (bright field vs. fluorescence).  

![alt text](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/data_bowl_data.PNG)



#### This algorithm needs to identify a range of nuclei across varied conditions. 

Except from the segmentetion problem, there is another problem:  

The images data were acquired under a variety of conditions and vary in the cell type, magnification, and imaging modality (bright field vs. fluorescence).  

The images can be in RGB, RGBA and gray scale format, based on the modality in which they were acquired. For color images, a third dimension encodes the "channel" (e.g. Red, Green, and Blue).  

### This means that a learning model (svm, deep learning etc.) for this problem should be enquired with a pipeline to process each image to an appropriate input and output for it.  

## A review of basic techniques in image segmentation

**image segmentetion** is the process of partitioning a digital image into multiple segments (sets of pixels, a.k.a super-pixels)[1] The goal of segmentation is to simplify and/or change the representation of an image into something that is more meaningful and easier to analyze. Image segmentation is typically used to locate objects and boundaries (lines, curves, etc.) in images.   
 
### More precisely, image segmentation is the process of assigning a label to every pixel in an image such that pixels with the same label share certain characteristics.

##### basic techniques
The simplest method of image segmentation is called the **thresholding method**. [2][1]
This method is based on a clip-level (or a threshold value) to turn a gray-scale image into a binary image
The key of this method is to select the threshold value (or values when multiple-levels are selected). 
Several popular methods are used in industry including the maximum entropy method, Otsu's method (maximum variance), and k-means clustering.

![alt text](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/%E2%80%8F%E2%80%8Fotsu_threshold.PNG)

The **K-means algorithm**[3] is an iterative technique that is used to partition an image into K clusters.
In this case, distance is the squared or absolute difference between a pixel and a cluster center.   
The difference is typically based on pixel color, intensity, texture, and location, or a weighted combination of these factors.   
K can be selected manually, randomly, or by a heuristic.   
This algorithm is guaranteed to converge, but it may not return the optimal solution.   
The quality of the solution depends on the initial set of clusters and the value of K.  

![alt text](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/kmeans_segmentetion.PNG)

The **watershed transformation**[4] considers the gradient magnitude of an image as a topographic surface.
Pixels having the highest gradient magnitude intensities (GMIs) correspond to watershed lines, which represent the region boundaries.   Water placed on any pixel enclosed by a common watershed line flows downhill to a common local intensity minimum (LIM).  
Pixels draining to a common minimum form a catch basin, which represents a segment.  

![alt text](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/watershed_segmentetion.PNG)

#### Trainable segmentation:
Most segmentation methods are based only on color information of pixels in the image. Humans use much more knowledge than this when doing image segmentation, but implementing this knowledge would cost considerable computation time and would require a huge domain knowledge database, which is currently not available. In addition to traditional segmentation methods, there are trainable segmentation methods which can model some of this knowledge.

## Deep Learning consepts:
Deep learning is a subfield of machine learning. While both fall under the broad category of artificial intelligence, deep learning is what powers the most human-like artificial intelligence.    
Though the main ideas behind deep learning have been in place for decades, it wasn’t until data sets became large enough and computers got fast enough that their true power could be revealed.  

A good way to understand deep learning is to take a look at logistic regression:    
Logistic regression uses a binary classification on input data,  
the model takes the input's n features and uses a weighted sum over them, the weighted sum is then passed as an input to a log function  and the classification is activated to one if the log output is greater than a certen threshold.  
#### Logistic regression is a simple neural network.
![alt text](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/%E2%80%8F%E2%80%8FLR.PNG)

### Multi Layer Perceptron (MLP):
MLPs are just logistic regression where a set of nonlinear features are automatically learned from data.  
MLP is defined by several parameters: 
 - Number of hidden units in each layer
 - Number of hidden layers in the network
 - The nonlinear activation function: (could also be RELU an rectified linear unit or tanh)
 - Learning rate for to use in SGD (using the chain rule of derivatives)
 
![alt text](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/MLP.PNG)
 
### Training NN (neural network) or Logistic regression:
Logistic regression uses binary crossentropy as a loss function, which is a very popular technique in binary classification.  
When we train a model we are trying to minimize the loss function to get the model optimal weights, one way to minimize this loss function is using Gradient descent.

#### Gradient Descent (GD): [7]
Gradient descent is an optimization algorithm, where after each epoch (= pass over the training dataset) the model weights are updated incrementally.  
The magnitude and direction of the weight update is computed by taking a step in the opposite direction of the cost gradient, which is the derivative calculation of the loss function.  
The weights are updated according to the learning rate after each epoch.  

#### Stochastic Gradient Descent (SGD):
Stochastic gradient descent compute the cost gradient based on a single training sample and not the complete training set like regular gradient descent.  
In case of very large datasets, using GD can be quite costly.  
the term "stochastic" comes from the fact that a single training sample is a "stochastic approximation" of the "true" cost gradient.  
There are different tricks to improve the GD-based learning, one is choosing a decrease constant d that shrinks the learning rate over time.  
another is to learn momentum by adding a factor of the previous gradient to the weight update for faster updates.

#### Mini-batch Gradient Descent:
instead of computing the gradient from 1 sample or all n training samples: Mini-batch gradient Descent  update the model based on smaller groups of training samples.


## Convolutional Neural Network (CNN) 
Convolutional networks were inspired by biological processes in that the connectivity pattern between neurons resembles the organization of the animal visual cortex.[8]

A successfully neural network for image and text recognition required all neurons to be connected, resulting in an overly-complex   network structure and very long training times.   
The convolution operation brings a solution to this problem as it reduces the number of free parameters, each neuron is connected to only a small region of the input volume.   
The extent of this connectivity is a hyperparameter called the receptive field of the neuron. allowing the network to be deeper with fewer parameters.  
Yann LeCun from Facebook’s AI Research group built the first Convolution Neural Network in 1988 called LeNet.
![alt text](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/CNN.PNG)[9]
#### Convolutional
convolution is a mathematical operation on two functions (f and g) to produce a third function that expresses how the shape of one is modified by the other.  
Convolutional layers apply a convolution operation to the input with a filter (weights) on its receptive field, passing the result to the next layer.  
Convolution as a property of being translational invariant: The output signal strength is not dependent on where the features are located, but simply whether the features are present.  

#### Pooling
Combine the outputs of neuron clusters at one layer into a single neuron in the next layer.  
For example, max pooling uses the maximum value from each of a cluster of neurons at the prior layer (another example is using the average value from each of the clusters).  

#### Fully connected
Fully connected layers connect every neuron in one layer to every neuron in another layer.  
It is in principle the same as the traditional multi-layer perceptron neural network (MLP).

#### Weights
CNNs share weights in convolutional layers, which means that the same filter is used for each receptive field in the layer, this reduces memory footprint and improves performance.

A classic architecture for CNN:  
##### imput -> Conv -> Relu -> Conv -> Relu -> Pool -> Conv -> Relu -> Pool -> Fully Connected

## Use of Deep Learning in segmentation problem
One of the popular initial deep learning approaches was patch classification where each pixel was separately classified into classes using a patch of image around it.[10]  
Main reason to use patches was that classification networks usually have full connected layers and therefore required fixed size images.

In 2014, Fully Convolutional Networks (FCN) by Long et al. from Berkeley, popularized CNN architectures for dense predictions without any fully connected layers.  
This allowed segmentation maps to be generated for image of any size and was also much faster compared to the patch classification approach.  
Almost all the subsequent state of the art approaches on semantic segmentation adopted this paradigm.

Apart from fully connected layers, one of the main problems with using CNNs for segmentation is pooling layers.  
Pooling layers increase the field of view and are able to aggregate the context while discarding the ‘where’ information.  
However, semantic segmentation requires the exact alignment of class maps and thus, needs the ‘where’ information to be preserved. Two different classes of architectures evolved in the literature to tackle this issue.

### U-Net - TODO: add more information on u-net
Encoder gradually reduces the spatial dimension with pooling layers and decoder gradually recovers the object details and spatial dimension.  
There are shortcut connections from encoder to decoder to help decoder recover the object details better.   

![alt text](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/unet.png)

## Autoencoders

#### Latent variables: 
are variables that are not directly observed but are rather inferred (through a mathematical model) from other variables that are observed (directly measured).

#### Latent variable models: 
are mathematical models that aim to explain observed variables in terms of latent variables.
(Hidden Markov model, PCA, tc)

#### Hierarchical latent variable models: 
explain latent variables with latent variables
(Hierarchical Hidden Markov model, Trees, PCA, etc.)

### Learning Latent Variable Models
Use the Expectation-Maximization algorithm (Dempster,
Laird and Rubin, 1977)
I Goal is to find parameters θ that maximize the log
likelihood

## Ladder Networks
## Use of Ladder Network in a CNN
## Use of Ladder Network in segmentation problem
## Description of primary products

## Methods: 
_Detailed description of algorithms and computational models used,   
tools use,   
work process,   
work limitations _ 

## Bibliography.
[1] Linda G. Shapiro and George C. Stockman (2001): “Computer Vision”, pp 279-325, New Jersey, Prentice-Hall, ISBN 0-13-030796-3  
[2] https://www.mathworks.com/discovery/image-segmentation.html  
[3]  Barghout, Lauren; Sheynin, Jacob (2013). "Real-world scene perception and perceptual organization: Lessons from Computer Vision". Journal of Vision. 13 (9): 709–709. doi:10.1167/13.9.709.  
[4]  Olsen, O. and Nielsen, M.: Multi-scale gradient magnitude watershed segmentation, Proc. of ICIAP 97, Florence, Italy, Lecture Notes in Computer Science, pages 6–13. Springer Verlag, September 1997.  
[5] http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review  
[6]  Andrew L. Beam (a great introduction to deep learning): http://beamandrew.github.io/deeplearning/2017/02/23/deep_learning_101_part2.html
[7] Bottou, Léon (1998). "Online Algorithms and Stochastic Approximations". Online Learning and Neural Networks. Cambridge University Press. ISBN 978-0-521-65263-6
[8] https://en.wikipedia.org/wiki/Convolutional_neural_network
[9] http://neuralnetworksanddeeplearning.com/
[10] http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review
