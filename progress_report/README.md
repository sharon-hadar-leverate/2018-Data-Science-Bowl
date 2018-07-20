# Progress Report Introduction: 
This README file is a progress report for the project "2018-Data-Science-Bowl".    
In this project, I am going to explore the possibilities of deep learning and segmentation problem assuming that combining ladders networks will enhance the hidden network learning features.  
#### In this report i will introduce: 
- [X] The project description 
- [X] A brif review of basic techniques in image segmentation
- [ ] Deep Learning consepts
- [ ] Use of Deep Learning in segmentation problem
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

A good way to understand deep learning is to look at logistic regression:  
logistic regression uses binary classification on input data, 
the model takes the input's n fetures and use weighted average on them, then it uses a log function on the weighted average and uses a threshold to activate a classification as one.  
#### Logistic regression is a simple neural network.


#### multilayer perceptron



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





## Methods: 
_Detailed description of algorithms and computational models used,   
tools use,   
work process,   
work limitations _ 

## Description of primary products.

## Bibliography.
[1] Linda G. Shapiro and George C. Stockman (2001): “Computer Vision”, pp 279-325, New Jersey, Prentice-Hall, ISBN 0-13-030796-3  
[2] https://www.mathworks.com/discovery/image-segmentation.html  
[3]  Barghout, Lauren; Sheynin, Jacob (2013). "Real-world scene perception and perceptual organization: Lessons from Computer Vision". Journal of Vision. 13 (9): 709–709. doi:10.1167/13.9.709.  
[4]  Olsen, O. and Nielsen, M.: Multi-scale gradient magnitude watershed segmentation, Proc. of ICIAP 97, Florence, Italy, Lecture Notes in Computer Science, pages 6–13. Springer Verlag, September 1997.  
[5] http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review  
[6]  Andrew L. Beam (a great introduction to deep learning): http://beamandrew.github.io/deeplearning/2017/02/23/deep_learning_101_part2.html
