# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./sample_train_images/center_2019_01_21_00_08_01_353.jpg "Sample 1"
[image2]: ./sample_train_images/left_2019_01_21_00_08_11_848.jpg "Sample 2"


## Rubric Points

---
### Files Submitted & Code Quality
My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 64 in the function network

The model includes RELU layers to introduce nonlinearity , and the data is normalized in the model using a Keras lambda layer

- Batch Size = 32. 
- test_size=0.2
-  epochs=3

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually . Inside the function network()

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy
#### 1. Preprocessing
- Used all the set of random left , center and right images. 
- Fliped the images . 
- Changed the brightness of each selected image. 



#### 2. Solution Design Approach


My first step was to use a convolution neural network model similar to the Nvida model. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. Increase the epoch to 3 from 1. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py function network()) consisted of a convolution neural network with the following layers and layer sizes 

Layer						|     Output Shape          |  Param _________________________________________________________________
lambda_1 (Lambda)   		|    (None, 160, 320, 3) 	|  0         
_________________________________________________________________
cropping2d_1 (Cropping2D)   | 	(None, 65, 320, 3)      |  0         
_________________________________________________________________
conv2d_1 (Conv2D)           |	(None, 33, 160, 24)     |  1824      
_________________________________________________________________
conv2d_2 (Conv2D)           | (None, 17, 80, 36)        | 21636     
_________________________________________________________________
conv2d_3 (Conv2D)           | (None, 9, 40, 48)         | 43248     
_________________________________________________________________
conv2d_4 (Conv2D)           | (None, 7, 38, 64)         | 27712     
_________________________________________________________________
conv2d_5 (Conv2D)           | (None, 5, 36, 64)         | 36928     
_________________________________________________________________
flatten_1 (Flatten)         | (None, 11520)             | 0         
_________________________________________________________________
dense_1 (Dense)             | (None, 100)               | 1152100   
_________________________________________________________________
dense_2 (Dense)             | (None, 50)                | 5050      
_________________________________________________________________
dense_3 (Dense)             | (None, 10)                | 510       
_________________________________________________________________
dense_4 (Dense)             | (None, 1)                 | 11 
_________________________________________________________________
Total params: 1,289,019
Trainable params: 1,289,019
Non-trainable params: 0
_________________________________________________________________

#### 3. Creation of the Training Set & Training Process

For the final train model I used udacity provided data.
- My data consisted of 2 center lanes. 
- Curve lanes. 
- Zig Zag motioned moving.
My data was heavy enough and training took time. So went with udacity provided data. 