# Progress Report Introduction: 
In this work I will present my thesis that a Ladder network could improve the accuracy of existing methods for medical image segmentation.  

I devided this project into two parts: 
 - Part One: Existing Segmentetion Techniques - where i would build a benchmark of existing segmentetion technigues and models with reference to the "2018 data sience bowl" goals.  
 - Part Two: Use Of Ladder Network - where i would combine ladder network with a deep learning segmentetion state of the art model. 
 
In order to understand if ladder network helps improving segmentation, i will build a benchmark of various methods for segmentation with performance measures (as IOU).
I will start by reviewing what is image segmentation and what are the basic techniques in image segmentation,
I will present what are the image segmentation performance measures and practice a threshold technique to get the first row of my banchmark,  
Then i would review trainable segmentation techniques with deep learning: 
ill start with explaining basic consepts of deep learning, convolutional neural network (CNN) and the use of it in segmentation problem, 
i would present a second row in my benchmark which is a full connected network (FCN),  
Then, i will use Unet, which is a state of the art deep learning model for image segmentation, and add it as row to my benchmark

#### In this report i will introduce: 

### Part One:  :waxing_gibbous_moon:
- [X] 2018 data sience bowl description 
- [X] A brif review of basic techniques in image segmentation
- [X] Deep Learning consepts
- [X] Convolutional Neural Network (CNN)
- [X] Use of Deep Learning in segmentation problem
- [X] [Description of primary products](2018-Data-Science-Bowl.ipynb)
### Part Two: :waning_crescent_moon:
- [ ] Autoencoders
- [ ] Ladder Networks
- [ ] Use of Ladder Network in a CNN
- [ ] Use of Ladder Network in segmentation problem
- [ ] Description of primary products
- [ ] Bibliography

## 2018 data sience bowl description
_“2018 Data Science Bowl” is a Kaggle competition that its goal is to create an algorithm to automate nucleus detection in divergent images to advance medical discovery._  

_By observing patterns, asking questions, and building a model, participants will have a chance to push state-of-the-art technology farther._  

in this competition, the challanger expose a dataset contains a large number of segmented nuclei images.  
The images were acquired under a variety of conditions and vary in the cell type, magnification, and imaging modality (bright field vs. fluorescence).  
The images can be in RGB, RGBA and gray scale format, based on the modality in which they were acquired. For color images, a third dimension encodes the "channel" (e.g. Red, Green, and Blue). 

![alt text](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/data_bowl_data1.PNG)  
#### This algorithm needs to identify a range of nuclei across varied conditions.  
## A review of basic techniques in image segmentation

**image segmentetion** is the process of partitioning a digital image into multiple segments (sets of pixels, a.k.a super-pixels)[1] The goal of segmentation is to simplify and/or change the representation of an image into something that is more meaningful and easier to analyze. Image segmentation is typically used to locate objects and boundaries (lines, curves, etc.) in images.   
 
### More precisely, image segmentation is the process of assigning a label to every pixel in an image such that pixels with the same label share certain characteristics.

##### basic techniques
The simplest method of image segmentation is called the **thresholding method**. [2][1]
This method is based on a clip-level (or a threshold value) to turn a gray-scale image into a binary image
The key of this method is to select the threshold value (or values when multiple-levels are selected). 
Several popular methods are used in industry including the maximum entropy method, Otsu's method (maximum variance), and k-means clustering.

![Fotsu_threshold](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/%E2%80%8F%E2%80%8Fotsu_threshold.PNG)

The **K-means algorithm**[3] is an iterative technique that is used to partition an image into K clusters.
In this case, distance is the squared or absolute difference between a pixel and a cluster center.   
The difference is typically based on pixel color, intensity, texture, and location, or a weighted combination of these factors.   
K can be selected manually, randomly, or by a heuristic.   
This algorithm is guaranteed to converge, but it may not return the optimal solution.   
The quality of the solution depends on the initial set of clusters and the value of K.  

![kmeans_segmentetion](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/kmeans_segmentetion.PNG)

The **watershed transformation**[4] considers the gradient magnitude of an image as a topographic surface.
Pixels having the highest gradient magnitude intensities (GMIs) correspond to watershed lines, which represent the region boundaries.   Water placed on any pixel enclosed by a common watershed line flows downhill to a common local intensity minimum (LIM).  
Pixels draining to a common minimum form a catch basin, which represents a segment.  

![watershed_segmentetion](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/watershed_segmentetion.PNG)

#### Trainable segmentation:
Most segmentation methods are based only on color information of pixels in the image. Humans use much more knowledge than this when doing image segmentation, but implementing this knowledge would cost considerable computation time and would require a huge domain knowledge database, which is currently not available. In addition to traditional segmentation methods, there are trainable segmentation methods which can model some of this knowledge.

### Threshold exploring:
When using different threshold methods on a training sample, the following segmentations received:   
![thresholds](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/thresholds.png)  
Threshold Yen (implements thresholding based on a maximum correlation criterion for bilevel thresholding as a more computationally efficient alternative to entropy measures.[12]) seems to have the best IoU over explored thresholds for this task.   
In the figure below is the original nuclei images, the image segmentation (ground truth) and Yen thresholding (from left to right) 
![threshold Yen](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/threshold%20Yen.png)  
Threshold Yen got a nice segmentation with almost no flase positive with an avarage of 0.698 IoU over all test data   
### IoU - Intersection over union
IoU is a segmentation performance measure which stand for intersection over union.  
The intersection (A∩B) is comprised of the pixels found in both the prediction mask and the ground truth mask, whereas the union (A∪B) is simply comprised of all pixels found in either the prediction or target mask.  

![threshold Yen](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/IOU_TH_YEN.png)  
Intersection over union for this case:  
![threshold Yen](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/IOU_TH_YEN2.png)  

| technique   | IoU |
| ------------- | ------------- |
| Threshold Yen | 0.698  |





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

#### Adam Optimization Algorithm:
Adam is an optimization algorithm that can used instead of the classical stochastic gradient descent procedure to update network weights iterative based in training data, the algorithm calculates an exponential moving average of the gradient and the squared gradient.

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
![alt text](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/pixlewise.PNG)  

In 2014, Fully Convolutional Networks (FCN) by Long et al. from Berkeley, popularized CNN architectures for dense predictions without any fully connected layers.  
This allowed segmentation maps to be generated for image of any size and was also much faster compared to the patch classification approach.  
![alt text](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/FCN.PNG)  
Almost all the subsequent state of the art approaches on semantic segmentation adopted this paradigm.  
(pictures from Stanford University School of Engineering course)

Apart from fully connected layers, one of the main problems with using CNNs for segmentation is pooling layers.  
Pooling layers increase the field of view and are able to aggregate the context while discarding the ‘where’ information.  
However, semantic segmentation requires the exact alignment of class maps and thus, needs the ‘where’ information to be preserved.

### U-Net
Encoder gradually reduces the spatial dimension with pooling layers and decoder gradually recovers the object details and spatial dimension.  
There are shortcut connections from encoder to decoder to help decoder recover the object details better.   

![alt text](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/unet.png)  
[11] U-net architecture (example for 32x32 pixels in the lowest resolution).  
Each blue box corresponds to a multi-channel feature map.  
The number of channels is denoted on top of the box.  
The x-y-size is provided at the lower left edge of the box. White
boxes represent copied feature maps.  
The arrows denote the different operations. 

The u-net architecture achieves very good performance on very different biomedical segmentation applications
The network consist of 2 paths:
#### Contracting:
Convolutional network with two 3x3 convolutions followed by a ReLU activation function and 2x2 max pooling with stride of 2.   
At each downsampling step the network doubles the number of feature channels.  

#### Expansive:
Every step consists of an upsampling of the feature map followed by a 2x2 convolution that halves the number of feature channels.  
Also, to prevent the loss of border pixels, a concatenation with the correspondingly cropped feature map from the contracting path, 
Then, two 3x3 convolutions, followed by a ReLU.
The cropping is necessary due to the loss of border pixels in every convolution.

At the final layer a 1x1 convolution is used to map each 64-
component feature vector to the desired number of classes. In total the network
has 23 convolutional layers.

## Unet exploring:
In order to create a deep learning model i use'd keras over Tensorflow kernal.    
Keras is an open source neural network library written in Python.   
It is capable of running on top of TensorFlow, Microsoft Cognitive Toolkit or Theano.    
TensorFlow is an open-source software library for dataflow programming across a range of tasks.   
It is a symbolic math library, and is also used for machine learning applications such as neural networks.  
I implemented U-net NN based on the article "U-Net: Convolutional Networks for Biomedical Image Segmentation" (Olaf Ronneberger, Philipp Fischer, Thomas Brox).  
since in the next step i would combine Ladder network with Unet, ill use a smaller image input size.  
Instead of 256 by 256 pixle image ill reduce the size of an image to 128 by 128 pixles.    
The network new architecture is:


| Layer (type) | Output Shape | Param | Connected to |  
| --- | --- | --- | --- |
| input (InputLayer) | (None, 128, 128, 3) | 0 | +++ |  
| dropout (Dropout) | (None, 128, 128, 3) | 0 | input |  
| conv2d_1 (Conv2D) | (None, 128, 128, 4) | 112 | dropout |  
| conv2d_2 (Conv2D) | (None, 128, 128, 4) | 148 | conv2d_1 |  
| max_pooling2d_1 (MaxPooling2D) | (None, 64, 64, 4) | 0 | conv2d_2 |
| conv2d_3 (Conv2D) | (None, 64, 64, 8) | 296 | max_pooling2d_1 | 
| conv2d_4 (Conv2D) | (None, 64, 64, 8) | 584 | conv2d_3 |
| max_pooling2d_2 (MaxPooling2D) | (None, 32, 32, 8) | 0 | conv2d_4 |
| conv2d_5 (Conv2D) | (None, 32, 32, 16) | 1168 | max_pooling2d_2 | 
| conv2d_6 (Conv2D) | (None, 32, 32, 16) | 2320 | conv2d_5 |
| max_pooling2d_3 (MaxPooling2D) | (None, 16, 16, 16) | 0 | conv2d_6 |
| conv2d_7 (Conv2D) | (None, 16, 16, 32) | 4640 | max_pooling2d_3 |
| conv2d_8 (Conv2D) | (None, 16, 16, 32) | 9248 | conv2d_7 |
| dropout_2 (Dropout) | (None, 16, 16, 32) | 0 | conv2d | 
| up_sampling2d_1 (UpSampling2D) | (None, 32, 32, 32) | 0 | dropout_2 | 
| conv2d_9 (Conv2D) | (None, 32, 32, 16) | 2064 | up_sampling2d_1|
| concatenate_1 (Concatenate) | (None, 32, 32, 32) | 0 | conv2d_6 conv2d_9 |
| conv2d_10 (Conv2D) | (None, 32, 32, 16) | 4624 | concatenate_1 | 
| conv2d_11 (Conv2D) | (None, 32, 32, 16) | 2320 | conv2d_10 | 
| up_sampling2d_2 (UpSampling2D) | (None, 64, 64, 16) | 0 | conv2d_11 | 
| conv2d_12 (Conv2D) | (None, 64, 64, 8) | 520 | up_sampling2d_2 |
| concatenate_2 (Concatenate) | (None, 64, 64, 16) | 0 |conv2d_4 conv2d_12 |
| conv2d_13 (Conv2D) | (None, 64, 64, 8) | 1160 | concatenate_2 | 
| conv2d_14 (Conv2D) | (None, 64, 64, 8) | 584 | conv2d_13 |
| up_sampling2d_3 (UpSampling2D) | (None, 128, 128, 8) | 0 | conv2d_14 |
| conv2d_15 (Conv2D) | (None, 128, 128, 4) | 132 | up_sampling2d_3 | 
| concatenate_3 (Concatenate) | (None, 128, 128, 8) | 0 | conv2d_2 conv2d_15 | 
| conv2d_16 (Conv2D) | (None, 128, 128, 4) | 292 | concatenate_3 | 
| conv2d_17 (Conv2D) | (None, 128, 128, 4) | 148 | conv2d_16 | 
| conv2d_18 (Conv2D) | (None, 128, 128, 2) | 74 | conv2d_17 |
| conv2d_19 (Conv2D) | (None, 128, 128, 1) | 3 | conv2d_18 |

____________________________________________________________________________________________________

Total params: 30,437  
Trainable params: 30,437  
Non-trainable params: 0  
____________________________________________________________________________________________________


When using different threshold methods on a training sample, the following segmentations received:   
![thresholds](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/thresholds.png)  
Threshold Yen (implements thresholding based on a maximum correlation criterion for bilevel thresholding as a more computationally efficient alternative to entropy measures.[12]) seems to have the best IoU over explored thresholds for this task.   
In the figure below is the original nuclei images, the image segmentation (ground truth) and Yen thresholding (from left to right) 
![threshold Yen](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/threshold%20Yen.png)  
Threshold Yen got a nice segmentatiodn with almost no flase positive with an avarage of 0.698 IoU over all test data   

![threshold Yen](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/IOU_TH_YEN.png)  
Intersection over union for this case:  
![threshold Yen](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/IOU_TH_YEN2.png)  

| technique   | IoU |
| ------------- | ------------- |
| Threshold Yen | 0.698  |



## Description of primary products
[click here](2018-Data-Science-Bowl.ipynb) to see the hole jupyter notebook


--------------------------------------------PART TWO----------------------------------------------------------------------------
## Autoencoders
An autoencoder learns to compress data from the input layer into a short code, and then uncompress that code into something that closely matches the original data.  
The simplest form of an autoencoder is a feedforward neural network having an input layer where the output layer having the same number of nodes as the input layer, and with the purpose of reconstructing its own inputs (instead of predicting the target value Y given inputs X).   
Therefore, autoencoders are unsupervised learning models.  


![autoencoder2](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/autoencoder2.png)  


#### Denoising Autoencoders
This idea relay on the Hebbian learning concept - A synapse between two neurons is strengthened when the neurons on either side of the synapse (input and output) have highly correlated outputs.

Learn representation that would be robust to introduction of noise will enforce the hidden unit to extract particular types of structures of correlations and to learn the training data distribution and more meaningful features.

2 ways to use Hebbian learning in deep learning models:
 - Dropouts - Random assignment of subset of inputs to 0, with the probability of V.
 - Gaussian additive noise.
![denoise_autoencoder](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/denoise_autoencoder.png)  

Hidden layer representetion (what the encoder has learn by levels of corruption) 
![hidden_layer_rep_dae](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/hidden_layer_rep_dae.png)  
Each square is one hidden unit visualization of the weight vector between all inputs and the specific hidden unit [13]  
There is a more significant representation as the signal is more corrupt, clear edges of digits are shown at 50% corruption.  
  



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
[11] https://arxiv.org/abs/1505.04597
[12] https://ieeexplore.ieee.org/abstract/document/366472/
[13] https://www.youtube.com/watch?v=6DO_jVbDP3I&t=1s
