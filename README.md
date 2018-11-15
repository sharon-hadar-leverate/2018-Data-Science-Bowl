
# Find the nuclei in divergent images    
____________________________________________________________________________________________________  
![top](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/new_assets/%E2%80%8F%E2%80%8FTOP.PNG)  

## Abstract
Image segmentation is the process of assigning a label to each pixel in the image where each label is a group with certain characteristics, it is widely used in medical image processing, face recognition, autonomous vehicle etc.  
In the 2010s, Deep Learning revolution and the use of it for medical segmentation has gained momentum,[21] in March 2015, a convolutional encoder-decoder architecture called Unet got the highest rank on the EM segmentation challenge [22].  
Apart from segmentation, Deep Learning has improved the data latent variables learning with the use of Autoencoders, a Neural Network that tries to reconstruct is own input, the use of noisy input and batch normalization improve the ability of the network to learn meaningful futures.  
One example of this improvement is the 'Ladder Network', which resembles a stacked Denoising Autoencoders and received state of the art scores in classification problems.[14]   
In this work, 
I present my thesis that using noisy input and batch normalization could improve the accuracy of existing methods for medical image segmentation.  
In order to show any improvement, I built a benchmark of various methods for segmentation with IOU as a performance measure.  
I used Kaggle's “2018 Data Science Bowl” competition data and started the work in identifying and analyzing the data.  
I have summarized several methods for segmentation and present what are the image segmentation performance measures.      
I explain and show my experience with the threshold technique to get the first row in my benchmark,   
I have summarized several trainable segmentation techniques with deep learning:   
I explain some basic concepts of deep learning, convolutional neural network (CNN) and the use of it in segmentation problem,
I explain and show my experience with Fully Convolutional Networks (FCN) in order to get the second row in my benchmark, 
In addition, I explain the Unet model, which is a state of the art deep learning model for image segmentation, and present my experience with it in order to get the third row in my benchmark,  
After acquiring these models, I present their combination with noisy input and batch normalization and evaluate these models.

#### In this report i will introduce: 

- [X] Abstract
- [X] 2018 data sience bowl description 
- [X] Investigate the data
- [X] Image segmentation
- [X] Threshold as a segmentation technique
- [X] IoU - intersection over union (performance index)
- [X] Deep Learning and Neural Network consepts
- [X] Multi Layer Perceptron (MLP)
- [X] How to train a Neural Network
- [X] Convolutional Neural Network (CNN)
- [X] Use of Deep Learning in segmentation problem
- [X] Fully Convolutional Networks exploring
- [X] U-Net 
- [X] Unet exploring
- [X] Autoencoders
- [X] Batch Normalization
- [X] Discussion and future development
- [X] Related work
- [X] [Description of primary products](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/nuclei_segmentation.ipynb)
- [X] Figure list
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
> (Figure 1: images by segmentation and histogram. the first line is the images, the second line is the image segmentation and the last line is the image histogram)   

The images are clearly different, for example, we can see that the first image is grayscale where the third image is purple and light purple.  
  
Dimension reduction techniques can be used for better visualisation of the data: 
  
![image_embedding](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/new_assets/image_embedding.png)  
> (Figure 2: pca embedding (left) UMAP embedding (right))   

In this figure, two techniques were used:  
PCA which is a mathematical transformation from related variables into unrelated variables based on the variables largest possible variance,  

and UMAP (Uniform Manifold Approximation and Projection, Feb 2018)[18], Which is a new approach to reducing dimensions, using local approximation and various corrections, along with simple fuzzy local representations.

UMAP shows better visualization than PCA, also, according to UMAP paper, it is demonstrably faster than t-SNE and provides better scaling.   
  
<p align="center"><img width="460" height="300" src="https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/new_assets/UMAP_embedding_with_images.png"></p>   

> (Figure 3: umap embedding with annotated images, each image is annotated to her 2d umap projection)   

Clustering the data into groups can help identify the different groups of images in the data, a good unsupervised clustering method for this problem is DBSCAN.  
DBSCAN (Density-based spatial clustering of applications with noise) groups together points that are close to each other based on a distance measurement and a minimum number of points.  
DBSCAN finds the optimum number of clusters and does not need an input the number of clusters to generate [19].  
DBSCAN was also proven to be better than other clustering technics according to sklearn benchmark [20].

![image_groups_by_image](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/new_assets/image_groups_by_image.png)  
> (Figure 4: Images embedding groups, each column is a group (5), each shows 3 different sample in the group and the samples histograms)   

It seems that additional mining is required, for example, group number 3 includes purple and grayscale images.  
one exploring direction is to use the image histogram which gives an overall idea about the intensity distribution of an image,  
In the plot above, it seems that a grayscale image has a similar distribute to other grayscale images but has different distribute to purple images.  

<p align="center"><img width="600" height="300" src="https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/new_assets/UMAP_embedding_with_images_hist.png"></p>  

> (Figure 5: umap image histogram embedding with annotated images, each image is annotated to her 2d umap projection based on images histograms)   


![image_groups_by_hist](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/new_assets/image_groups_by_hist3.png) 
> (Figure 6: images histograms embedding groups, each column is a group (9), each shows 3 different sample in the group and the samples histograms)    

The clusters information:  

| group | #samples | background color | nuclei color | nuclei radios | nuclei amounth | precentage |
| ------------- | ------------- | -------------  | ------------- | ------------- | ------------- | ------------- |
| 0 | 257 | black | gray | small -> medium | medium | 45% |
| 1 | 36 | white | purple | small  | medium -> many | 6.3% |
| 2 | 30 | light perpule | perpule | large | few | 5.2% |
| 3 | 96 | black | gray -> white | extra small | few -> medium | 16.8% |
| 4 | 66 | black | gray -> white | extra small | a lot | 11.5 |
| 5 | 25 | black | gray | large | medium | 4.3% |
| 6 | 16 | white | gray | medium | medium |  2.8% |
| 7 | 32 | light perpule | perpule | extra small | medium -> many | 5.6% | 
| 8 | 12 | black | gray | small -> extra large | one -> few | 2.1% |

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
> (Figure 7: the arbitrary threshold, in this plot (from left to right), are the original image, a binarization of an image with an arbitrary threshold of 50, the image actual mask and the image histogram where the black line is the threshold)   

For example, this figure uses an arbitrary threshold of 50 on a random image,  
Each cell in the image has a value between 0 to 256, where 0 is black and 256 is white.  
The threshold methods assign a new value base on the original value.  
If the original value is above the threshold (>50) the value is assigned to be one (white), and if the value is below the threshold the value is assigned to be zero (black)

Several popular methods are used in industry including Otsu's method (maximum variance), and Yen method (maximum correlation).  

#### Otsu threshold: 
Otsu The most commonly used threshold segmentation algorithm that uses the largest interclass variance  which selects a globally optimal threshold by maximizing the variance between classes.[21]  

#### Yen threshold: 
In this method the threshold is calculated base on the incompatibility between the final image and the original image,  
Its implements thresholding based on a maximum correlation criterion for bilevel thresholding as a more computationally efficient alternative to entropy measures.[12].  
The threshold is calculated per image:  

![th_yen](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/new_assets/th_yen.png)  
> (Figure 8: Yen threshold example 1, in this plot (from left to right), are the original image, a binarization of an image with Yen threshold, the image actual mask and the image histogram where the black line is Yen threshold (11))   

The binarization should be reversed in cases where the nuclei is darker than the background,  
If the original value is **below** the threshold the value is assigned to be one (white), and if the value is **above** the threshold the value is assigned to be zero (black):  

![th_yen2](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/new_assets/th_yen2.png)  
> (Figure 9: Yen threshold example 2, in this plot (from left to right), are the original image, a binarization of an image with Yen threshold, the image actual mask and the image histogram where the black line is Yen threshold (164))   
   
### IoU - Intersection over union (performance index)
IoU is a segmentation performance measure which stands for intersection over union.  
The intersection (A∩B) is comprised of the pixels found in both the prediction mask and the ground truth mask, 
whereas the union (A∪B) is simply comprised of all pixels found in either the prediction or target mask.  
  
![iou1](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/new_assets/iou1.png)  
> (Figure 10: IOU example 1, in this plot (from left to right up to down), is a binarization of an image with Yen threshold, the image actual mask, the intersection between them and the union between them)   

Intersection over union for this case (where white is intersection and grey is union):  
  
![iou2](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/new_assets/iou2.png)  
> (Figure 11: IOU example 2, the Intersection over the union, the intersection is white and the union without intersection is grey, the Iou is the white are divided by the white and gray area)  

Choosing a threshold would directly impact the IoU score:  
  
![score_per_iou](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/new_assets/score_per_iou.png)  
> (Figure 12: IOU score per threshold, each row is the same sample that been examen by different threshold, the first column is the image histogram where the black line is Yen threshold, same is the figure above, and the red line is the exam threshold, the second column is the binarization of the image based on that threshold and the third column is the IOU plot)  

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
> (Figure 13: kmeans segmentetion)  

The **watershed transformation**[4] considers the gradient magnitude of an image as a topographic surface.
Pixels having the highest gradient magnitude intensities (GMIs) correspond to watershed lines, which represent the region boundaries.    Water placed on any pixel enclosed by a common watershed line flows downhill to a common local intensity minimum (LIM).  
Pixels draining to a common minimum form a catch basin, which represents a segment.  

![watershed_segmentetion](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/watershed_segmentetion.PNG)
> (Figure 14: watershed segmentetion) 

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
> (Figure 15: logistic regression, the input are the blue units, the summarization is the green sigma unit and the activation is the yellow unit) 

### Multi Layer Perceptron (MLP):
MLPs are simple neural networks in a stack, where one layers output is used as input to the next layer. 
MLP is defined by several parameters:  
 - Number of hidden units in each layer  
 - Number of hidden layers in the network  
 - The activation functions at each layer.
 
![alt text](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/MLP.PNG)  
> (Figure 16: mlp, the input are the units in the red rectangle, the hidden layer are each summarization and activation unit layers, the last unit is the output) 

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
> (Figure 17: Convolutional Neural Network (CNN), get an image as an input, then use convolution on a sliding window follow by max polling and a fully connected layer which classify the image into a bird, sunset or another label) 

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
> (Figure 18: dumb pixel Network, gets an image as an input, and extract patches to classify each of them to cow or grass) 

In 2014, Fully Convolutional Networks (FCN) by Long et al. from Berkeley, popularized CNN architectures for dense predictions without any fully connected layers.  
This allowed segmentation maps to be generated for an image of any size and was also much faster compared to the patch classification approach.  
![alt text](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/FCN.PNG)  
> (Figure 19: fcn, gets an image as an input, and use multi-layer convolution the use activate to each label at pixel resolution)  

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
> (Figure 20: fcn learning curves, train and test model accuracy by epochs (left) model loss by  epochs (right))  

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
> (Figure 21: fcn model evaluation, in this figure each column belongs to a different group in the data, the first row is the image, the second row is the image truth mask, the third row is the model projection before thresholding the fourth row is the binarization of the projection and the last row is the sample iou)  

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
> (Figure 22: u-net architecture,[11] Each blue box corresponds to a multi-channel feature map.  
The number of channels is denoted on top of the box.  
The x-y-size is provided at the lower left edge of the box.   
White boxes represent copied feature maps.   
The arrows denote the different operations.)  

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
| conv2d_1 (Conv2D) | (None, 128, 128, 4) | 2 | dropout |  
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
> (Figure 23: unet learning curves, train and test model accuracy by epochs (left) model loss by  epochs (right))  

When comparing Unet to FCN, Unet has 30,437 trainable parameters,   
almost 30 times more parameters then FCN, the training procedure takes more time and required more memory,   
the large model (which includes an additional contracting and expansive step) also require a strong GPU.  
``` 
mean test IOU:  0.8218275476316186
```  
![unet](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/new_assets/UNET_v2_evaluate.png)  
> (Figure 24: unet model evaluation, in this figure each column belongs to a different group in the data, the first row is the image, the second row is the image truth mask, the third row is the model projection before thresholding the fourth row is the binarization of the projection and the last row is the sample iou) 

Unet got a highest average IoU score of 0.825.  

| technique   | Mean IoU |
| ------------- | ------------- |
| UNET | 0.821 |
| FCN | 0.738 |
| Threshold Otsu | 0.718 |
| Threshold Yen | 0.696 |

This is the first part of acquiring a baseline. in the next part, I would combine these methods with noisy input and batch normalization.   
In order to understand the usage of noisy input and batch normalization, one need to understand Denoising Autoencoders.  

## Autoencoders

An autoencoder learns to encode the input layer into a shortcode, and then decode the shortcode to closely matches the original input.   
The simplest form of an autoencoder is a feedforward neural network having an input layer where the output layer having the same number of nodes as the input layer,   
and with the purpose of reconstructing its own inputs (instead of predicting the target value Y given inputs X).  
Therefore, autoencoders are unsupervised learning models.  

#### UNET learns to encode the input layer into a shortcode, and then decode the shortcode to closely matches the original input segmentation. 

#### Denoising Autoencoders  
This idea relay on the Hebbian learning concept - A synapse between two neurons is strengthened when the neurons on either side of the synapse (input and output) have highly correlated outputs.  

Learn representation that would be robust to the introduction of noise will enforce the hidden unit to extract particular types of structures of correlations and to learn the training data distribution and more meaningful features.

2 ways to use Hebbian learning in deep learning models:
 - Dropouts - Random assignment of subset of inputs to 0, with the probability of V.
 - Gaussian additive noise.  

The introduction of noise causes the sample to distance itself from the data distribution.  
Then, when learning to reconstruct the same sample without the noise, the gradient is forced to contain a component that is precisely oriented to the data distribution.  
![denoise_autoencoder](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/denoise_autoencoder.png)  
> (Figure 25: denoise autoencoder, in a projection of abstract demensions) 

Hidden layer representation (what the encoder has learn by levels of corruption)  
![hidden_layer_rep_dae](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/hidden_layer_rep_dae.png)
> (Figure 26: hidden layer mean activation by denoise (corruption rates)) 

Each square is one hidden unit visualization of the weight vector between all inputs and the specific hidden unit [13]  
There is a more significant representation as the signal is more corrupt, clear edges of digits are shown at 50% corruption.

Random Gaussian noise was added to the train set with mean 0 and SD of 256:  
![adding_noise](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/new_assets/adding_noise.png )  
> (Figure 27: noise input, the first row is the original image, the second row is the original image histogram, the third row is the image corrupt, the fourth row is the corrupted image histogram and the last row is the image mask) 

When training a new Unet model on the noised data, the mean iou is not improved:    
```
mean test IOU:  0.820656430606358
```  
![denoised lc](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/new_assets/denoised.png)  
> (Figure 28: denoised unet learning curves, train and test model accuracy by epochs (left) model loss by  epochs (right))  

![noisy large unet v3](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/new_assets/denoised1.png)  
> (Figure 29: denoised unet model evaluation, in this figure each column belongs to a different group in the data, the first row is the image, the second row is the image truth mask, the third row is the model projection before thresholding the fourth row is the binarization of the projection and the last row is the sample iou)  

### Batch Normalization  
To increase the stability of a neural network, batch normalization normalizes the output of a previous activation layer by subtracting the batch mean and dividing by the batch standard deviation.
It is a technique to provide any layer in a neural network with inputs that are zero mean/unit variance.
If an algorithm learned some X to Y mapping, and if the distribution of X changes, then we might need to retrain the learning algorithm by trying to align the distribution of X with the distribution of Y.[23]

When adding batch normalization to Unet model per convolution layer on the noised data, the mean iou is improved:  
```
mean test IOU:  0.8343306901825481
```  
![bn noisy large unet v3](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/new_assets/denoising_bn_UNET_v2_evaluate.png)  
> (Figure 30: denoised unet with batch normalization model evaluation, in this figure each column belongs to a different group in the data, the first row is the image, the second row is the image truth mask, the third row is the model projection before thresholding the fourth row is the binarization of the projection and the last row is the sample iou)   

| technique   | Mean IoU |
| ------------- | ------------- |
| Noisy UNET with Batch Normalization   | 0.834 |
| UNET with Batch Normalization  | 0.828 |
| UNET | 0.821 |
| Noisy UNET | 0.820 |
| FCN | 0.738 |
| Threshold Otsu | 0.718 |
| Threshold Yen | 0.696 |


## Discussion and future development:  

In this project, I introduced the idea that using noisy input and batch normalization could improve the accuracy of existing methods for medical image segmentation.   
In order to show any improvement, I built a benchmark of various methods for segmentation with IOU as a performance measure.  
I used Kaggle's “2018 Data Science Bowl” and after analyzing it I have shown that it is mainly characterized by the number of nuclei in an image, the nucleus width, and the image colors, and also that the data can be gathered into different groups that could receive different treatment.  
This benchmark has both classic segmentation technics and modern trainable segmentation technics, it is clearly shown that using trainable modern technics, as deep learning, perform better than classic technics, as Threshold Otsu\ Yen.  
This benchmark also shows that Unet receives better IOU score than Fully Convolutional Networks, which was expected since it is state of the art technic for medical image segmentation.  

The combination of Batch normalization with the Unet model received better scores than the Unet along, 
The data was trained mainly by grayscale images (80%) and when applying Unet network to colored images the performance gained less Iou score, the training set, and the prediction set are both nucleus images but they differ, Unet learns some encoded mapping of different groups of images, and if the distribution of one group has changed, then we might need to retrain the learning algorithm by trying to align the distribution of this group with the distribution of other groups, which is exactly what batch normalization does.  
The introduction to noise improved the Iou score when combining it with Batch normalization, noise made the model learn connections between groups, while this along did improve the Iou of weaker groups in the data (for example, a sample from group #2 received 0.758 Iou on the regular Unet and 0.84 Iou on the Unet model with noise), the overall Iou score wasn't better, this could imply that combining Batch normalization with noise also improve the model global learning compare to a noise alone Unet.    
 
Future development is to adjust the Unet architecture further and combine it with the Ladder Network architecture,   
Ladder Network, received state of the art results, learns the latent space of each layer by combining denoising autoencoders to each layer and adding the autoencoders loss function to the network cost function, the combination could be that each downsampling in the model would become an autoencoder.  
Combining U-net with these components isn't trivial and could create a very complex model, 
one that will probably require more memory and GPU.

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

## Description of products
[click here](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/nuclei_segmentation.ipynb) to see the hole jupyter notebook

#### List of figures:
- figure 1: images by segmentation and histogram.
- figure 2: PCA embedding and UMAP embedding
- figure 3: UMAP embedding with annotated images
- figure 4: Images embedding groups
- figure 5: UMAP image histogram embedding with annotated images
- figure 6: Images histograms embedding groups,
- figure 7: The arbitrary threshold
- figure 8: Yen threshold example 1
- figure 9: Yen threshold example 2
- figure 10 IOU example 1
- figure 11: IOU example 2
- figure 12: IOU score per threshold
- figure 13: Kmeans segmentetion
- figure 14: Watershed segmentetion
- figure 15: Logistic regression
- figure 16 MLP
- figure 17: Convolutional Neural Network (CNN)
- figure 18: Dumb pixel Network
- figure 19: FCN
- figure 20: FCN learning curves
- figure 21: FCN model evaluation
- figure 22: Unet architecture
- figure 23: Unet learning curves
- figure 24: Unet model evaluation
- figure 25: Denoise Autoencoder
- figure 26: Hidden layer mean activation by denoise
- figure 27: Noise input
- figure 28: Denoised Unet learning curves
- figure 29: Denoised Unet model evaluation
- figure 30: Denoised Unet with Batch Normalization model evaluation  

## Bibliography.
[1] Linda G. Shapiro and George C. Stockman (2001): “Computer Vision”, pp 279-325, New Jersey, Prentice-Hall, ISBN 0-13-030796-3  
[2] https://www.mathworks.com/discovery/image-segmentation.html   MathWorks "Segmentation methods in image processing and analysis"
[3]  Barghout, Lauren; Sheynin, Jacob (2013). "Real-world scene perception and perceptual organization: Lessons from Computer Vision". Journal of Vision. 13 (9): 709–709. doi:10.1167/13.9.709.  
[4]  Olsen, O. and Nielsen, M.: Multi-scale gradient magnitude watershed segmentation, Proc. of ICIAP 97, Florence, Italy, Lecture Notes in Computer Science, pages 6–13. Springer Verlag, September 1997.   
[5] http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review  
[6]  Andrew L. Beam (a great introduction to deep learning):   http://beamandrew.github.io/deeplearning/2017/02/23/deep_learning_101_part2.html  
[7] Bottou, Léon (1998). "Online Algorithms and Stochastic Approximations". Online Learning and Neural Networks. Cambridge University Press. ISBN 978-0-521-65263-6  
[8] https://en.wikipedia.org/wiki/Convolutional_neural_network  "Convolutional neural network"
From Wikipedia, the free encyclopedia  
[9] http://neuralnetworksanddeeplearning.com/  "Deep Learning", book by Ian Goodfellow, Yoshua Bengio, and Aaron Courville  
[10] http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review Sasank Chilamkurthy (July 5, 2017), "A 2017 Guide to Semantic Segmentation with Deep Learning"  
[11] https://arxiv.org/abs/1505.04597  Olaf Ronneberger, Philipp Fischer, Thomas Brox (18 May 2015) "U-Net: Convolutional Networks for Biomedical Image Segmentation".  
[12] https://ieeexplore.ieee.org/abstract/document/366472/  Jui-Cheng Yen, Fu-Juay Chang, Shyang Chang ( Mar 1995) "A new criterion for automatic multilevel thresholding"  
[13] https://www.youtube.com/watch?v=6DO_jVbDP3I&t=1s  Hugo Larochelle (nov 2013) "Neural networks [6.3] : Autoencoder - example"   
[14] https://arxiv.org/pdf/1507.02672.pdf  Antti Rasmus, Harri Valpola, Mikko Honkala, The Curious AI Company, Finland (jul 2015) "Semi-Supervised Learning with Ladder Networks"    
[15] https://engineering.purdue.edu/elab/CortexNet/ Alfredo Canziani, Eugenio Culurciello (Jun 2017) "CortexNet: a robust predictive deep neural network trained on videos"  
[16] https://arxiv.org/pdf/1707.03718.pdf Abhishek Chaurasia, Eugenio Culurciello (Jun 2017) "LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation"   
[17] https://arxiv.org/pdf/1606.06724.pdf Antti Rasmus, Harri Valpola, Mikko Honkala, The Curious AI Company, Finland (nov 2016) "Tagger: Deep Unsupervised Perceptual Grouping"  
[18] https://arxiv.org/pdf/1802.03426.pdf Leland McInnes and John Healy (feb 2018) "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction"  
[19] https://web.cs.dal.ca/~zincir/bildiri/jias-ckdnm.pdf Carlos Bacquet1, Kubra Gumus2, Dogukan Tizer3, A. Nur Zincir-Heywood4 and Malcolm I. Heywood5 "A Comparison of Unsupervised Learning Techniques for Encrypted Traffic Identification"  
[20] http://scikit-learn.org/0.15/auto_examples/cluster/plot_cluster_comparison.html#example-cluster-plot-cluster-comparison-py sklearn "Comparing different clustering algorithms on toy datasets"  
[21] https://arxiv.org/ftp/arxiv/papers/1707/1707.02051.pdf Song Yuheng1, Yan Hao1 "Image Segmentation Algorithms Overview"
[22] http://brainiac2.mit.edu/isbi_challenge/ "ISBI Challenge: Segmentation of neuronal structures in EM stacks"
[23] https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c Firdaouss Doukkal (oct 2017) "Batch normalization in Neural Networks"

