## Introduction: 
_A short description of the subject and its main purpose._

“2018 Data Science Bowl” is a Kaggle competition that its goal is to create an algorithm to automate nucleus detection in divergent images to advance medical discovery.  
The Data This dataset contains a large number of segmented nuclei images.

## The problem 
_discussed by the work the engineering challenge._

image segmentetion: In computer vision, image segmentation is the process of partitioning a digital image into multiple segments (sets of pixels, a.k.a super-pixels)[1] The goal of segmentation is to simplify and/or change the representation of an image into something that is more meaningful and easier to analyze. Image segmentation is typically used to locate objects and boundaries (lines, curves, etc.) in images.   
More precisely,   
### image segmentation is the process of assigning a label to every pixel in an image such that pixels with the same label share certain characteristics.

Except from the segmentetion problem, there is another problem:  

The images data were acquired under a variety of conditions and vary in the cell type, magnification, and imaging modality (bright field vs. fluorescence).  

The images can be in RGB, RGBA and gray scale format, based on the modality in which they were acquired. For color images, a third dimension encodes the "channel" (e.g. Red, Green, and Blue).  

### This means that a learning model (svm, deep learning etc.) for this problem should be enquired with a pipeline to process each image to an appropriate input and output for it.  


## Expanded literature review:  
_The status of current scientific knowledge on the topic of the thesis,   
comparison The current thesis for similar works_  

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
