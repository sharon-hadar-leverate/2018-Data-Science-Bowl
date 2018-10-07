
# Progress Report Introduction: 
____________________________________________________________________________________________________
21.08.2018  
Find the nuclei in divergent images to advance medical discovery  
מציאת גרעיני תא בתמונות מסתעפות לקדם גילוי רפואי
 
Student Sharon Hadar   
Supervisor Sharon Yalov-handzel, DR   
____________________________________________________________________________________________________


In this work I will present my thesis that a Ladder network could improve the accuracy of existing methods for medical image segmentation.  


In order to understand if the ladder network helps improving segmentation, 
I will build a benchmark of various methods for segmentation with performance measures (as IOU).   
I will start by reviewing what is image segmentation and what are the basic techniques in image segmentation,   
I will present what are the image segmentation performance measures and practice a threshold technique to get the first row of my benchmark,   
Then I would review trainable segmentation techniques with deep learning:   
Ill start with explaining basic concepts of deep learning,
convolutional neural network (CNN) and the use of it in segmentation problem,    
I would present the second row in my benchmark which is a  Fully Convolutional Networks (FCN),  
Then, I will use Unet, which is a state of the art deep learning model for image segmentation, 
and add it as a row to my benchmark.   
After acquiring a baseline with two neural networks (a simple and a complex one) 
I would combine each with a ladder network components and evaluate these models.  

#### In this report i will introduce: 

- [X] 2018 data sience bowl description 
- [X] A brif review of basic techniques in image segmentation
- [X] Deep Learning consepts
- [X] Convolutional Neural Network (CNN)
- [X] Use of Deep Learning in segmentation problem
- [X] Autoencoders
- [X] Ladder Networks
- [X] Future Work
- [X] [Description of primary products](2018-Data-Science-Bowl.ipynb)
- [X] Bibliography

## 2018 data sience bowl description
“2018 Data Science Bowl” is a Kaggle competition that its goal is to create an algorithm to automate nucleus detection in divergent images to advance medical discovery.  

_By observing patterns, asking questions, and building a model, participants will have a chance to push state-of-the-art technology farther._  

In this competition, the challenger exposes a dataset contains a large number of segmented nuclei images.  
The images were acquired under a variety of conditions and vary in the cell type, magnification, and imaging modality (bright field vs. fluorescence).  
The images can be in RGB, RGBA and grayscale format, based on the modality in which they were acquired. For color images, a third dimension encodes the "channel" (e.g. Red, Green, and Blue).  


#### This algorithm needs to identify a range of nuclei across varied conditions.  


## Investigate The Data

Basic information:  
* Number of train samples:  570.   
* Number of test samples:  100.  
* Each image is reshaped to (128,128,3)  

In order to understand the data we first need to look at some random images.  
  
![plot_images](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/new_assets/plot_images.png)  
The images are clearly different, for example, we can see that the first image is grayscale where the third image is purple and light purple.  
  
Dimension reduction techniques can be used for better visualisation of the data: 
  
![image_embedding](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/new_assets/image_embedding.png)  
In this figure, two techniques were used:  
PCA which is a mathematical transformation from related variables into unrelated variables based on the variables largest possible variance,  

and UMAP (Uniform Manifold Approximation and Projection, Feb 2018)[18], Which is a new approach to reducing dimensions, using local approximation and various corrections, along with simple fuzzy local representations.

UMAP shows better visualization than PCA, also, according to UMAP paper, it is demonstrably faster than t-SNE and provides better scaling.   
  
<p align="center"><img width="460" height="300" src="https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/new_assets/UMAP_embedding_with_images.png"></p>   

Clustering the data into groups can help identify the different groups of images in the data, a good unsupervised clustering method for this problem is DBSCAN.  
DBSCAN (Density-based spatial clustering of applications with noise) groups together points that are close to each other based on a distance measurement and a minimum number of points.  
DBSCAN finds the optimum number of clusters and does not need an input the number of clusters to generate [19].  
DBSCAN was also proven to be better than other clustering technics according to sklearn benchmark [20].

![image_groups_by_image](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/new_assets/image_groups_by_image.png)  

It seems that additional mining is required, for example, group number 3 includes purple and grayscale images.  
one exploring direction is to use the image histogram which gives an overall idea about the intensity distribution of an image,  
In the plot above, it seems that a grayscale image has a similar distribute to other grayscale images but has different distribute to purple images.  

<p align="center"><img width="600" height="300" src="https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/new_assets/UMAP_embedding_with_images_hist.png"></p>  

![image_groups_by_hist](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/new_assets/image_groups_by_hist3.png)  

The clusters information:  

| group | #samples | background color | nuclei color | nuclei radios | nuclei amounth | 
| ------------- | ------------- | -------------  | ------------- | ------------- | ------------- |
| 0 | 257 | black | gray | small -> medium | medium |
| 1 | 36 | white | purple | small  | medium -> many |
| 2 | 30 | light perpule | perpule | large | few |
| 3 | 96 | black | gray -> white | extra small | few -> medium |
| 4 | 66 | black | gray -> white | extra small | a lot | 
| 5 | 25 | black | gray | large | medium | 
| 6 | 16 | white | gray | medium | medium | 
| 7 | 32 | light perpule | perpule | extra small | medium -> many | 
| 8 | 12 | black | gray | small -> extra large | one -> few | 

For conclusion, the data are mainly characterized by the number of nuclei in an image, the nucleus width, and the image colors.  
The data can be gathered into different groups that could receive different treatment.  
Most of the data is grayscale with a small amount of medium size nucleus.  
The data minority group has a gray nucleus with a white background, which is the opposite of most of the data.   


## Image Segmentation

Image Segmentation is the process of partitioning a digital image into multiple segments (sets of pixels, a.k.a super-pixels)[1]   
The goal of segmentation is to simplify and/or change the representation of an image into something that is more meaningful and easier to analyze.   
Image segmentation is typically used to locate objects and boundaries (lines, curves, etc.) in images.  

### More precisely, image segmentation is the process of assigning a label to every pixel in an image such that pixels with the same label share certain characteristics.

## Threshold as a segmentation technique

Threshold method for image segmentation is a binarization of the image according to a selected threshold.  
  
![arbitrary_th](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/new_assets/arbitrary_th.png)  
For example, this figure uses an arbitrary threshold of 50 on a random image,  
Each cell in the image has a value between 0 to 256, where 0 is black and 256 is white.  
The threshold methods assign a new value base on the original value.  
If the original value is above the threshold (>50) the value is assigned to be one (white), and if the value is below the threshold the value is assigned to be zero (black)

Several popular methods are used in industry including Otsu's method (maximum variance), and Yen method (maximum correlation).  

#### Yen threshold: 
In this method the threshold is calculated base on the incompatibility between the final image and the original image,  
Its implements thresholding based on a maximum correlation criterion for bilevel thresholding as a more computationally efficient alternative to entropy measures.[12].  
The threshold is calculated per image:  

![th_yen](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/new_assets/th_yen.png)  

The binarization should be reversed in cases where the nuclei is darker than the background,  
If the original value is **below** the threshold the value is assigned to be one (white), and if the value is **above** the threshold the value is assigned to be zero (black):  

![th_yen2](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/new_assets/th_yen2.png)  
   
### IoU - Intersection over union (performance index)
IoU is a segmentation performance measure which stands for intersection over union.  
The intersection (A∩B) is comprised of the pixels found in both the prediction mask and the ground truth mask, 
whereas the union (A∪B) is simply comprised of all pixels found in either the prediction or target mask.  
  
![iou1](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/new_assets/iou1.png)  

Intersection over union for this case (where white is intersection and grey is union):  
  
![iou2](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/new_assets/iou2.png)  

Choosing a threshold would directly impact the IoU score:  
  
![score_per_iou](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/new_assets/score_per_iou.png)  
A reasonable value for a good segmentation is above 0.7.  
0.4 IOU score is considered to be poor segmentation.
  
The benchmark performance would be measured by mean IOU score on the test data.  
Both threshold methods score:  

| technique   | Mean IoU |
| ------------- | ------------- |
| Threshold Otsu | 0.718 |
| Threshold Yen | 0.696 |


##### Additional Segmentation Techniques

The **K-means algorithm**[3] is an iterative technique that is used to partition an image into K clusters.
In this case, distance is the squared or absolute difference between a pixel and a cluster center.   
The difference is typically based on pixel color, intensity, texture, and location, or a weighted combination of these factors.   
K can be selected manually, randomly, or by a heuristic.   
This algorithm is guaranteed to converge, but it may not return the optimal solution.   
The quality of the solution depends on the initial set of clusters and the value of K.  

![kmeans_segmentetion](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/kmeans_segmentetion.PNG)

The **watershed transformation**[4] considers the gradient magnitude of an image as a topographic surface.
Pixels having the highest gradient magnitude intensities (GMIs) correspond to watershed lines, which represent the region boundaries.    Water placed on any pixel enclosed by a common watershed line flows downhill to a common local intensity minimum (LIM).  
Pixels draining to a common minimum form a catch basin, which represents a segment.  

![watershed_segmentetion](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/watershed_segmentetion.PNG)

#### Trainable segmentation:
Most segmentation methods are based only on color information of pixels in the image. Humans use much more knowledge than this when doing image segmentation, but implementing this knowledge would cost considerable computation time and would require a huge domain knowledge database, which is currently not available.   
In addition to traditional segmentation methods, there are trainable segmentation methods which can model some of this knowledge.  
One of these methods is Deep learning.  


## Deep Learning and Neural Network consepts:

**Deep learning** is a subfield of machine learning algorithms that examine and construct models of hierarchical representation of the data, deep learning is what powers the most human-like (natural) artificial intelligence like image and speech recognizion.  
Though the main ideas behind deep learning have been in place for decades, it wasn’t until data sets became large enough and computers got fast enough that their true power could be revealed.  
  
**Neural networks** simulates a lot of tightly connected units in order to describe a hierarchical model, learn to recognize patterns and make decisions in a human manner.   

#### A neural network does not need to be programmed to study explicitly, it learns everything by itself.   

A neural network is characterized by:  
**Architecture:** Its pattern of connections between the neurons.  
**Activation function:** Neurons get activated if the network input exceeds their threshold value.  
**Learning algorithm:** Its method of determining the weights on the connections.  
  
One way to understand neural networks is to observe logistic regression:  
Logistic regression uses a binary classification on input data,  
Architecture: the model takes the input's n features and uses a weighted sum over them,   
Activation function: the weighted sum is then passed as an input to a log function and the classification is activated to one if the log output is greater than a certain threshold.   
Learning algorithm: learn optimal weights by using optimization techniques such as gradient descent.    
#### Logistic regression is a simple neural network.
![alt text](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/%E2%80%8F%E2%80%8FLR.PNG)  


### Multi Layer Perceptron (MLP):
MLPs are simple neural networks in a stack, where one layers output is used as input to the next layer. 
MLP is defined by several parameters:  
 - Number of hidden units in each layer  
 - Number of hidden layers in the network  
 - The activation functions at each layer.
 
![alt text](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/MLP.PNG)
  
### How To Train a Neural Network:    
When we train a model we are trying to minimize the loss function to get the model optimal weights,  
for example, Logistic regression uses binary cross-entropy as a loss function, which is a very popular technique in binary classification.  
one way to minimize this loss function is using Gradient Descent.  

#### Gradient Descent (GD): [7]

Gradient descent is an optimization algorithm, where after each epoch (= pass over the training dataset) the model weights are updated incrementally.   
The magnitude and direction of the weight update are computed by taking a step in the opposite direction of the cost gradient, which is the derivative calculation of the loss function.  
The weights are updated according to the learning rate after each epoch.  

#### Stochastic Gradient Descent (SGD):
Stochastic gradient descent computes the cost gradient based on a single training sample and not the complete training set like regular gradient descent.  
In the case of very large datasets, using GD can be quite costly.  
the term "stochastic" comes from the fact that a single training sample is a "stochastic approximation" of the "true" cost gradient.  
There are different tricks to improve the GD-based learning, one is choosing a decrease constant d that shrinks the learning rate over time.  
another is to learn momentum by adding a factor of the previous gradient to the weight update for faster updates.  

#### Mini-batch Gradient Descent:
Instead of computing the gradient from 1 sample or all n training samples:   
Mini-batch gradient Descent updates the model based on smaller groups of training samples.

#### Adam Optimization Algorithm:  
Adam is an optimization algorithm that can be used instead of the classical stochastic gradient descent procedure to update network weights iterative based in training data, 
the algorithm calculates an exponential moving average of the gradient and the squared gradient.  


A successfully neural network for image and text recognition required all neurons to be connected, resulting in an overly-complex   network structure and very long training times.   

## Convolutional Neural Network (CNN)   
The convolution operation brings a solution to this problem as it reduces the number of free parameters, each neuron is connected to only a small region of the input volume.   
The extent of this connectivity is a hyperparameter called the receptive field of the neuron.   
allowing the network to be deeper with fewer parameters.  
Convolutional networks were inspired by biological processes in that the connectivity pattern between neurons resembles the organization of the animal visual cortex.[8]  
Yann LeCun from Facebook’s AI Research group built the first Convolution Neural Network in 1988 called LeNet. [9]
![alt text](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/CNN.PNG)  

The CNN components are:  

#### Convolutional
convolution is a mathematical operation on two functions (f and g) to produce a third function that expresses how the shape of one is modified by the other.  
Convolutional layers apply a convolution operation to the input with a filter (weights) on its receptive field, passing the result to the next layer.  
Convolution as a property of being translational invariant:  
The output signal strength is not dependent on where the features are located, but simply whether the features are present.   

#### Pooling
Combine the outputs of neuron clusters at one layer into a single neuron in the next layer.   
For example, max pooling uses the maximum value from each of a cluster of neurons at the prior layer (another example is using the average value from each of the clusters).   

#### Fully connected
Fully connected layers connect every neuron in one layer to every neuron in another layer.   
It is in principle the same as the traditional multi-layer perceptron neural network (MLP). 

#### Weights
CNNs share weights in convolutional layers, which means that the same filter is used for each receptive field in the layer, 
this reduces memory footprint and improves performance.

A classic architecture for CNN:  
##### imput -> Conv -> Relu -> Conv -> Relu -> Pool -> Conv -> Relu -> Pool -> Fully Connected

## Use of Deep Learning in segmentation problem

One of the popular initial deep learning approaches was patch classification where each pixel was separately classified into classes using a patch of the image around it.[10]   
The main reason to use patches was that classification networks usually have full connected layers and therefore required fixed size images.   
![alt text](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/pixlewise.PNG)  


In 2014, Fully Convolutional Networks (FCN) by Long et al. from Berkeley, popularized CNN architectures for dense predictions without any fully connected layers.  
This allowed segmentation maps to be generated for an image of any size and was also much faster compared to the patch classification approach.  
![alt text](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/FCN.PNG)  

Almost all the subsequent state of the art approaches on semantic segmentation adopted this paradigm.  
(pictures from Stanford University School of Engineering course)

## Fully Convolutional Networks exploring:

In order to create a deep learning model, I used keras over Tensorflow kernel.    
Keras is an open source neural network library written in Python.    
It is capable of running on top of TensorFlow, Microsoft Cognitive Toolkit or Theano.  
TensorFlow is an open-source software library for dataflow programming across a range of tasks.  
It is a symbolic math library and is also used for machine learning applications such as neural networks.  

I implemented one variant of an FCN without pulling layer to test a base score for deep learning and trainable model:  

| Layer (type) | Output Shape | Param | Connected to |  
| --- | --- | --- | --- |
| input (InputLayer) | (None, 128, 128, 3) | 0 | +++ |  
| conv2d_1 (Conv2D) |  (None, 128, 128, 8) | 224 | input |  
| dropout (Conv2D) | (None, 128, 128, 8) | 0 | conv2d_1 |  
| conv2d_2 (Conv2D) | (None, 128, 128, 8) | 584 | dropout |  
| conv2d_3 (Conv2D) | (None, 128, 128, 4) | 292 | conv2d_2 |
| conv2d_4 (Conv2D) | (None, 128, 128, 2) | 74  | conv2d_3 | 
| conv2d_5 (Conv2D) | (None, 128, 128, 1) | 3 | conv2d_4 |

____________________________________________________________________________________________________

Total params: 1,177  
Trainable params: 1,177  
Non-trainable params: 0  
____________________________________________________________________________________________________


I tried different number of epochs, learning rates, and batch sizes on this architecture and evaluate them base on the test set mean iou.  
the full investigation is shown in this jupyter notebook.  
the most successful parameters are:  
epochs: 100 with early stop - meaning that the training stops after 5 epochs with no change in the val_loss.  
batch_size: of 8 images.    
learning rate: 1e-3.  

![FCN_v2_lcurves](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/new_assets/FCN_v2_lcurves.png)

The learning curve is sharp at first but becomes monotonous rather quickly, the monotonous slope indicates slow learning that requires a greater amount of data, an arithmetic that does not require a large amount of data has a higher learning curve and becomes very quickly blunt.  

The model prediction has one channel and a spectrum of values while the segmentation mask has only 2 values.
In order to get a mask prediction, I used binarization with a threshold. 
I also tried using Yen thresholding which received mean iou of 0.55 and Otsu thresholding which is better with mean iou of 0.66,   
But the best resolved received using a threshold by testing all train set samples iou's as a function of all thresholds from 0 to 1 with a step of 0.001.
The result:  
``` 
mean test IOU:  0.7381347267693156
```  
![FCN_vs_evaluate](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/new_assets/FCN_vs_evaluate.png)  

Different data groups got different treatment with different iou scores,   
the first and fourth groups, which is most of the data, got pretty good iou, while other groups got below 0.72 and even close to zero.   
FCN received an IoU average 0.738, which is better than the Threshold Otsu:  

| technique   | Mean IoU |
| ------------- | ------------- |
| FCN | 0.738 |
| Threshold Otsu | 0.718 |
| Threshold Yen | 0.696 |

Apart from fully connected layers, one of the main problems with using CNN's for segmentation is pooling layers.  
Pooling layers increase the field of view and are able to aggregate the context while discarding the ‘where’ information.    
However, semantic segmentation requires the exact alignment of class maps and thus, needs the ‘where’ information to be preserved.   
U-net is a good architecture to tackle this issue.  

### U-Net
In 2015, Olaf Ronneberger, Philipp Fischer and Thomas Brox proposed a new deep learning architecture for image segmentation called the U-net.  
Encoder gradually reduces the spatial dimension with pooling layers and decoder gradually recovers the object details and spatial dimension.  
There are shortcut connections from the encoder to the decoder to help decoder recover the object details better.  
![alt text](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/unet.png)  

[11] U-net architecture (an example of 32x32 pixels in the lowest resolution).  
Each blue box corresponds to a multi-channel feature map.  
The number of channels is denoted on top of the box.  
The x-y-size is provided at the lower left edge of the box.   
White boxes represent copied feature maps.   
The arrows denote the different operations.  

The u-net architecture achieves very good performance on very different biomedical segmentation applications.   
The network consist of 2 paths:  

#### Contracting:
Convolutional network with two 3x3 convolutions followed by a ReLU activation function and 2x2 max pooling with a stride of 2.  
At each downsampling step, the network doubles the number of feature channels.  

#### Expansive:
Every step consists of an upsampling of the feature map followed by a 2x2 convolution that halves the number of feature channels.  
Also, to prevent the loss of border pixels,
a concatenation with the correspondingly cropped feature map from the contracting path,  
Then, two 3x3 convolutions, followed by a ReLU.  
The cropping is necessary due to the loss of border pixels in every convolution.  

At the final layer, 
a 1x1 convolution is used to map each 64- component feature vector to the desired number of classes.  
In total the network has 23 convolutional layers.  

## Unet exploring:

I implemented U-net NN based on the article "U-Net: Convolutional Networks for Biomedical Image Segmentation" (Olaf Ronneberger, Philipp Fischer, Thomas Brox).  
since in the next step, I would combine Ladder network with Unet, ill use a smaller image input size.  
Instead of 256 by 256-pixel image, I will reduce the size of an image to 128 by 128 pixels.  
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

![large_unet_v2_lcurves](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/new_assets/large_unet_v2_lcurves.png)   

When comparing Unet to FCN, Unet has 30,437 trainable parameters,   
almost 30 times more parameters then FCN, the training procedure takes more time and required more memory,   
the large model (which includes an additional contracting and expansive step) also require a strong GPU.  
``` 
mean test IOU:  0.8218275476316186
```  
![unet](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/new_assets/UNET_v2_evaluate.png)  
Unet got a highest average IoU score of 0.825.  

| technique   | Mean IoU |
| ------------- | ------------- |
| UNET | 0.821 |
| FCN | 0.738 |
| Threshold Otsu | 0.718 |
| Threshold Yen | 0.696 |

This is the first part of acquiring a baseline. in the next part, I would combine these methods with Ladder Networks components.    
In order to understand Ladder Network, one need to understand Denoising Autoencoders.  

## Autoencoders

An autoencoder learns to encode the input layer into a shortcode, and then decode the shortcode to closely matches the original input.   
The simplest form of an autoencoder is a feedforward neural network having an input layer where the output layer having the same number of nodes as the input layer,   
and with the purpose of reconstructing its own inputs (instead of predicting the target value Y given inputs X).  
Therefore, autoencoders are unsupervised learning models.  

#### UNET learns to encode the input layer into a shortcode, and then decode the shortcode to closely matches the original input segmentetion.  

#### Denoising Autoencoders  
This idea relay on the Hebbian learning concept - A synapse between two neurons is strengthened when the neurons on either side of the synapse (input and output) have highly correlated outputs.  

Learn representation that would be robust to the introduction of noise will enforce the hidden unit to extract particular types of structures of correlations and to learn the training data distribution and more meaningful features.

2 ways to use Hebbian learning in deep learning models:
 - Dropouts - Random assignment of subset of inputs to 0, with the probability of V.
 - Gaussian additive noise.  

The introduction of noise causes the sample to distance itself from the data distribution.  
Then, when learning to reconstruct the same sample without the noise, the gradient is forced to contain a component that is precisely oriented to the data distribution.  
![denoise_autoencoder](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/denoise_autoencoder.png)  

Hidden layer representation (what the encoder has learn by levels of corruption)  
![hidden_layer_rep_dae](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/hidden_layer_rep_dae.png)

Each square is one hidden unit visualization of the weight vector between all inputs and the specific hidden unit [13]  
There is a more significant representation as the signal is more corrupt, clear edges of digits are shown at 50% corruption.

Random Gaussian noise was added to the train set with mean 0 and SD of 256:  
![adding_noise](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/new_assets/adding_noise.png )  

When training a new Unet model on the noised data, the mean iou is not improved:    
```
mean test IOU:  0.820656430606358
```  
![denoised lc](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/new_assets/denoised.png)  

![noisy large unet v3](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/new_assets/denoised1.png)  

### Batch Normalization  
TODO : write on BN from ladder network and remove ladder network  
TODO : show the new architecture


When adding batch normalization to Unet model per convolution layer on the noised data, the mean iou is improved:  
```
mean test IOU:  0.8343306901825481
```  
![bn noisy large unet v3](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/new_assets/denoising_bn_UNET_v2_evaluate.png)  

| technique   | Mean IoU |
| ------------- | ------------- |
| Noisy UNET with Batch Normalization   | 0.834 |
| UNET | 0.821 |
| Noisy UNET | 0.820 |
| FCN | 0.738 |
| Threshold Otsu | 0.718 |
| Threshold Yen | 0.696 |



TODO: add Ladder Networks and Unsupervised Pretraining to Related Work

#### Unsupervised Pretraining
One way to use the benefits of Denoising Autoencoders is by pretraining.  
Deep neural network model weights start with random values and 'twiks' with each sample/batch according to the gradient,   
when pretraining a deep neural network, 
the model start with almost optimal weights for the problem and is fine-tuned with the training samples.  
The idea is to use denoising autoencoder to train one layer at a time from the first hidden layer to the last, 
previous layers viewed as feature extraction when Fix the parameters of previously hidden layers:  
![pretrained](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/pretrain.png)   

 First layer: find hidden unit features that are more common in training inputs than in random input  
 Second layer: find combinations of hidden unit features that are more common than random hidden unit features.  
 Third layer: find combinations of combinations of..   


## Ladder Networks  [14]

There is a problem with pretraining: In complex tasks, there is often much more structure in the inputs than can be represented, 
and unsupervised learning cannot, by definition, know what will be useful for the task at hand.  
One instance is the autoencoder approach applied to natural images:   
The autoencoder will try to preserve all the details needed for reconstructing the image at a pixel level,  even though classification is typically invariant to all kinds of transformations which do not preserve pixel values.  
Most of the information required for pixel-level reconstruction is irrelevant and takes space from the more relevant invariant features which cannot alone be used for reconstruction.  

#### Ladder Network learns the latent space of each layer by combining denoising autoencoders to each layer and adding the autoencoders loss function to the network cost function.     
Ladder Network is a semi-supervised that was developed by "Harri Valpola" from "The Curious AI Company" back in nov 2015. The Curious AI Company was founded by Harri Valpola in 2015 in Helsinki,  
The company focuses on semi-supervised and unsupervised machine learning, which takes the human brain as its model.  

#### The structure of the Ladder network:  
![ladder network](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/ladder_network2.png)  
ladder network architecture resemble the unet architecture, These models use an encoder and a decoder pair to segment images into parts and objects.
f(l) The feedforward path with the corrupted feedforward path  
g(l) Denoising functions  
C(l)d Cost functions on each layer trying to minimize the difference between zˆ(l) and z(l)  



### Related Work:

Ladder networks and U-net inspired lots of researches in the field, 
one example is the CortexNet, which is one of the most complex deep neural network architectures, 
where adding recursion to ladder networks in order to learn videos segmentation frame by frame. [15]  
Another one is LinkNet, 
which resemble the U-net structure, 
composed by an encoder-decoder pair with skip connections between them, 
it was designed for real-time object segmentation, especially for self-driving cars, 
and needs to process images fast, that is why LinkNet has less polling and up-sampling layers then U-net.[16]  
The Curious AI Company extend the use of ladder networks concepts with a new framework for learning efficient iterative inference of perceptual grouping which called iTerative Amortized Grouping (TAG) where training can be completely unsupervised and additional terms for supervised tasks can be added too.   
Divide the input into K parts insert it to a recurrent ladder network, 
with denoising autoencoders, 
where each group is processed separately and learns its own prediction, 
in each iteration, save the group assignment probabilities and the expected value of the input for that group, 
and insert it ass an additional input to the next iteration. [17]  

### Future Work  
In the next step, I will add Batch Normalization and denoising autoencoder to the FCN and Unet models I implemented.  
Combining U-net with these components isn't trivial and could create a very complex model, 
one that will probably require more memory and GPU, 
in this case, I would use Google Cloud Platform compute engine \ ML engine.  
Finally, I will evaluate the new models in relation to the same baseline.  


## Description of primary products
[click here](2018-Data-Science-Bowl.ipynb) to see the hole jupyter notebook

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
[14] https://arxiv.org/pdf/1507.02672.pdf  
[15] https://engineering.purdue.edu/elab/CortexNet/  
[16] https://arxiv.org/pdf/1707.03718.pdf  
[17] https://arxiv.org/pdf/1606.06724.pdf  
[18] https://arxiv.org/pdf/1802.03426.pdf
[19] https://web.cs.dal.ca/~zincir/bildiri/jias-ckdnm.pdf
[20] http://scikit-learn.org/0.15/auto_examples/cluster/plot_cluster_comparison.html#example-cluster-plot-cluster-comparison-py
