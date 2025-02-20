# Brain-Tumor-Detection-CNN

## Motivation:
Brain tumor is one of the most dangerous diseases which require early and accurate detection methods. Now most detection and diagnosis methods depend on decision of neurospecialists, and radiologist for image evaluation which is prone to human errors and is time consuming. The main purpose of this project is to build a robust Convolution Neural Network model that can classify if the subject has a tumor or not based on brain MRI scan images with an acceptable accuracy for medical grade application.

## Domain related background:
A brain tumor is a mass or growth of abnormal cells in your brain.
Many different types of brain tumors exist. Some brain tumors are noncancerous (benign), and some brain tumors are cancerous (malignant). Brain tumors can begin in your brain (primary brain tumors), or cancer can begin in other parts of your body and spread to your brain (secondary, or metastatic, brain tumors).
How quickly a brain tumor grows can vary greatly. The growth rate as well as location of a brain tumor determines how it will affect the function of your nervous system.
Brain tumor treatment options depend on the type of brain tumor you have, as well as its size and location.

## Methodology:
## The Dataset:
A brain MRI images dataset founded on Kaggle. You can find it here. 
https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection 
The dataset contains 2 folders: yes and no which contains 253 Brain MRI Images. The folder yes contains 155 Brain MRI Images that are tumorous (malignant) and the folder no contains 98 Brain MRI Images that are non-tumorous (benign).
Meaning that 61% (155 images) of the data are positive examples and 39% (98 images) are negative.

## Data Preparation:
### Data augmentation 
As limited data was available, we have used data augmentation, a strategy that enables practitioners to significantly increase the diversity of data available for training models, without collecting new data.
Data augmentation techniques such as cropping, padding, and horizontal flipping are used and implemented using the ImageDataGenerator provided by Keras.
Before data Augmentation:
It consists of 253 brain MRI image, 155 were brain tumour positive and 98 were negative.
After data augmentation:
It consists of 2064 brain MRI image, 1084 were brain tumour positive and 980 were negative.
I have augmented in such a way to balance the dataset.

### Data Pre-processing:
For every image, the following pre-processing steps were applied:
1.	Crop the part of the image that contains only the brain (which is the most important part of the image): The cropping technique is used to find the extreme top, bottom, left and right points of the brain using OpenCV.
2.	Resize the image to have a shape of (240,240,3)
= (image_width, image_height, number of channels): because images in the dataset come in different sizes. So, all images should have the same shape to feed it as an input to the neural network.
3.	Apply normalization: to scale pixel values to the range 0–1.


![image](https://user-images.githubusercontent.com/46301535/118350084-8360f200-b572-11eb-9762-9cb0a8a2dff9.png)

### Data Split:
The data was split in the following way:
70% of the data for training.
15% of the data for validation (development).
15% of the data for testing.

## CODE FLOW BLOCK DIAGRAM

![image](https://user-images.githubusercontent.com/46301535/118350117-adb2af80-b572-11eb-9517-38de61f0a15b.png)

## The CNN Model
A Convolutional Neural Network (CNN) model is found to be the best suited approach for the problem statement. It is comprised of one or more convolution layers (often with a sub-sampling step) and then followed by one or more fully connected layers as in a standard multi-layer neural network. The main goal of the convolutional base is to generate features from the image. The architecture of a CNN is designed to take advantage of the 2D structure of an input image.
Understanding the architecture:
Each input x (image) has a shape of (240, 240, 3) and is fed into the neural network. And, it goes through the following layers:
1.	A Zero Padding layer 
2.	A convolutional layer with 32 filters, with a filter size of (7, 7) and a stride equal to 1.
3.	A batch normalization layer to normalize pixel values to speed up computation.
4.	A ReLU (Rectified Linear Unit) activation layer.
5.	A Max Pooling layer.
6.	A Max Pooling layer with f=4 and s=4, same as before. 
7.	A Flatten layer in order to flatten the 3-dimensional matrix into a one-dimensional vector.
8.	A Dense (output unit) fully connected layer with one neuron with a sigmoid activation (since this is a binary classification task).

![image](https://user-images.githubusercontent.com/46301535/118350143-d470e600-b572-11eb-85d9-09083eed7d98.png)

### Training the Model
The model was trained for 25 epochs and these are the loss & accuracy plots:

![image](https://user-images.githubusercontent.com/46301535/118350156-e3579880-b572-11eb-9ae9-9e405de2d347.png)

![image](https://user-images.githubusercontent.com/46301535/118350163-e9e61000-b572-11eb-9985-486a6cda224f.png)

### Results
Now, the best model (the one with the best validation accuracy) detects brain tumor with:
#### 93.5% accuracy on the test set.
#### 0.93 F1 score on the test set.

### Integrating the deep learning model with a web app to run on local host using Flask
![image](https://user-images.githubusercontent.com/46301535/118350193-0124fd80-b573-11eb-8174-26247c852ea5.png)

![image](https://user-images.githubusercontent.com/46301535/118350200-05e9b180-b573-11eb-9682-2b9d1cd242b4.png)




