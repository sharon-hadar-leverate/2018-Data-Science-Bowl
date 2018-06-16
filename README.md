#### “2018 Data Science Bowl” is a Kaggle competition that its goal is to create an algorithm to automate nucleus detection in divergent images to advance medical discovery. 

The Data This dataset contains a large number of segmented nuclei images. 
The images were acquired under a variety of conditions and vary in the cell type, magnification, and imaging modality (bright field vs. fluorescence).  

The images can be in RGB, RGBA and gray scale format, based on the modality in which they were acquired. For color images, a third dimension encodes the "channel" (e.g. Red, Green, and Blue).

### Description of work stages 
A successful type of models for image analysis are convolutional neural networks. 

Part one - Building a base line with MLP\CNN network: 

![alt text](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/part-one1.PNG)
![alt text](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/part-one2.PNG)

Part two – using a ladder network: 

![alt text](https://github.com/sharon-hadar-leverate/2018-Data-Science-Bowl/blob/master/assets/part-two.PNG)

My Assumption is that combining ladders networks would improve the network-hidden features learning 
 
