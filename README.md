# Udacity_SDC_P3_Behavioral-Cloning

[//]: # (Image References)
[image1]: ./images/Architecture.PNG "architecture"
[image2]: ./images/resizing_and_cropping.png "Resized and crooped image"

This is the third project for Udacity Self-Driving Car Engineer Nanodegree. For this project, we need to control a car in a simulator to collect camera images and steering angles, and then using this information to build a model to predict the steering angles based on the recorded camera images. The model learns from the data generated from the way we drive the car, hence the name.

The projects includes the following files:
* model.py - this is the file to create and train model. See the comments in the the file for explanation  
* model.h5 - a traned cnn for predicting the steering angle from the camera images 
* drive.py - for driving the car autonomously in the simulator. Added a function to resize the input images
* README.md - the file reading now
* test_recording.mp4 - a video which shows the car driving autonomously by the trained cnn model

Make sure Opencv is installed before running the script.

### Running the code
To test the model, launch the Udacity simulator and execute
```
python drive.py model.h5
```
Make sure Opencv is installed before running the script

## Model architecture
The archtecture is a convolutional neural netowrk based on the architecture of VGG net. In order to be able to train the model in a fair amount of time, I started from a smaller network and gradually expand the nodes for each layer and the number of layers until the model can drive around the track successfully. One interesting thing I found is that the number of filter is crucial for the car to make a big turn. The fianl archtecture is shown below:

<img src="./images/Architecture.PNG" width=500/>

It is an eight-layers convolunal networtk. The first part is two 5x5x32 convolutional layers followed by a 2x2 maxpooling layer, and the second part is two 3x3x64 convolutional layers followed by a 2x2 maxpooling layer. The final part is two fully-connected layers. ReLU is the activation function for each layer, and the dropout is used for reducing the overfitting. 

visualization of cnn output

## Training strategy
The training data was collected by driving around the track in the designated direction for three laps and one lap in the opposite direction. The number of total training sample is 10512. The data provided by Udacity was used for validation, which has 8036 instances.

The size of the original camera images are 160x320x3. The images are resized to 96x96x3 and further cropped to 55x96x3 when feeding into the network.

![][image2]

 images the quality describe how you do augmentaion and resampling

batch size and epochs and adam






