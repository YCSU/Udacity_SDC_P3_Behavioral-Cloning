# Udacity_SDC_P3_Behavioral-Cloning
This is the third project for Udacity Self-Driving Car Engineer Nanodegree. For this project, we need to control a car in a simulator to collect camera images and steering angles, and then using this information to build a model to predict the steering angles based on the recorded camera images. The model learns from the data generated from the way we drive the car, hence the name.

The projects includes the following files:
* model.py - this is the file to create and train model. See the comments in the the file for explanation  
* model.h5 - a traned cnn for predicting the steering angle from the camera images 
* drive.py - for driving the car autonomously in the simulator
* README.md - the file reading now
* test_recording.mp4 a video which shows the car driving autonomously by the trained cnn model

### Running the code
To test the model, launch the Udacity simulator and execute
```
python drive.py model.h5
```

### Model architecture and training strategy










