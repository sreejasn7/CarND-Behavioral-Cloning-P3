import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten , Dense , Lambda , Cropping2D , Conv2D,Dropout
import matplotlib.pyplot as plt
from keras.layers.pooling import MaxPooling2D
import sklearn 
from sklearn.model_selection import train_test_split

FILE_PATH = "/opt/carnd_p3/data/driving_log.csv"
IMAGE_PATH = "/opt/carnd_p3/data/IMG/"


#FILE_PATH = "/opt/carnd_p3/train_img/driving_log.csv"
#IMAGE_PATH = "/opt/carnd_p3/train_img/IMG/"

#load CSV files
def read_csv(filename):
    lines = []
    with open(filename) as file:
        reader = csv.reader(file)
        for line in reader:
            lines.append(line)
    
    
    return lines[1:]
    #return lines

# shuffle and get train and validation images. 
#Load the images . 

def read_images(line_paths):
    """ Get X_train and y_train """
    images = []
    steer_angles = []
    
    for line in line_paths:
        for i in range(3):
            source_path = line[i]
            file_name = source_path.split("/")[-1]
            current_path = IMAGE_PATH + file_name 
            image = cv2.imread(current_path)
            images.append(image)
            #angle = float(line[3])
            steer_angles.append(float(line[3]))
    X_train  = np.array(images)
    y_train = np.array(steer_angles)
    return X_train , y_train 

def augment_set(X, y):
    aug_img = []
    aug_angle = []
    for im , ang in zip(X ,y):
        aug_img.append(im)
        aug_angle.append(ang)
        aug_img.append(cv2.flip(im , 1))
        aug_angle.append(ang * -1.0)
    
    X_train = np.array(aug_img)
    y_train = np.array(aug_angle)
    return X_train , y_train
def read_sample_image(line_paths):
    """Read Sample Images - For testing purpose only """
    #for line in line_paths:
    source_path = line_paths[1]
    file_name = source_path.split("/")[-1]
    print(file_name)
    current_path = IMAGE_PATH + file_name 
    print(current_path)
    image = cv2.imread(current_path)
    return image
        
        


def network(image_shape,train_generator , validation_generator,train_samples,validation_samples):
    " " " Network to train model Model.h5 " " "
    model = Sequential()
    # Mean centering and Normaization.
    #ch, row, col = 3, 80, 320  # Trimmed image format
    model.add(Lambda(lambda x: ((x  / 255.0) - 0.5 ),input_shape=image_shape ))
    model.add(Cropping2D(cropping=((70 , 25) , (0,0))))
    model.add(Conv2D(24,(5,5) ,strides=(2,2) , padding="same", activation = "relu"))
    #model.add(MaxPooling2D())
    model.add(Conv2D(36,(5,5) ,strides=(2,2) , padding="same" , activation = "relu"))
    model.add(Conv2D(48,(5,5),strides=(2,2) , padding="same", activation = "relu"))
    model.add(Conv2D(64,(3,3), activation = "relu"))
    model.add(Conv2D(64,(3,3),activation = "relu"))
    #model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.2))

    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    #model.fit(X_train , y_train,validation_split=0.2,shuffle=True,nb_epoch=5)
    model.fit_generator(train_generator, steps_per_epoch= len(train_samples),validation_data=validation_generator, validation_steps=len(validation_samples), epochs=1, verbose = 1)

    model.summary()
    model.save("model.h5")
    
    

def change_img_brigtness(image):
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    bias = 0.25
    bright = bias + np.random.uniform()
    hsv[:,:,2] = hsv[:,:,2 ] * bright
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    top = 60
    bottom = 20
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                img_index = np.random.randint(3)
                source_path = batch_sample[img_index]
                file_name = source_path.split("/")[-1]
                current_path = IMAGE_PATH + file_name 

                img = cv2.imread(current_path)
                ang = float(batch_sample[3])

                #crop_img = img[top:img.shape[0] - bottom,:]
                #img = change_img_brigtness(img)
                images.append(img)
                angles.append(ang)

                images.append(cv2.flip(img , 1))
                angles.append(ang * -1.0)

            # trim image to only see section with road
            
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
    
# Main Page Execution sequence
if  __name__ == "__main__":
    # TRAINING
    lines = read_csv(FILE_PATH)
    print("lenth of lines" , len(lines))
    #print(lines[6000])
    img = read_sample_image(lines[1])
    img_shape = img.shape # (160, 320, 3)
    #print(img_shape)
    #print(lines[1])
    #img = read_sample_image(lines[1])
    #img_shape = img.shape # (160, 320, 3)
    print(img_shape)


    #-------------WITHOUT GENERATOR---------------
    #X_set, y_set = read_images(lines)
    #print(X_set.shape , y_set.shape)
    #X_train , y_train = augment_set(X_set  , y_set)
    #print(X_train.shape , y_train.shape)
    #network(img_shape , X_train , y_train)

    #-------------WITHOUT GENERATOR---------------
    
    # ------- WITH GENERATOR---------------
    train_samples, validation_samples = train_test_split(lines, test_size=0.2)
    print("train test split done") 
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)
    print("train generated ") 
    network(img_shape , train_generator , validation_generator,train_samples,validation_samples)
    print("Nework trained") 



    
    
    