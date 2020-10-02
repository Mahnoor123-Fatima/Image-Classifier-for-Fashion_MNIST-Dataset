[overview](#Project-Overview)
# Image-Classifier-for-Fashion_MNIST-Dataset
Build an image classifier using deep learning CNN for [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset.


# Description

Fashion training set consists of 70,000 images divided into 60,000 training and 10,000 testing samples. Dataset sample consists of 28x28 grayscale image, associated with a label from 10 classes.

The 10 classes are as follows:

0  T-shirt/top<br>
1  Trouser<br>
2  Pullover<br>
3  Dress<br>
4  Coat<br>
5  Sandal<br>
6  Shirt<br>
7  Sneaker<br>
8  Bag<br>
9  Ankle boot

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255.

# Project Overview

This project is divided into 10 parts; they are:

## 1 Importing of Dataset:<br>
```python
from tensorflow.keras.datasets import fashion_mnist
```
## 2 Loading of Data:<br>
```python
(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()
```
## 3 Preprocessing of data:<br>
- **Normalization of data**
```python
x_train = x_train/255 
x_test = x_test/255
```
- **reshape dataset to have a single channel**
```python
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
```
- **one hot encode target values**
```python
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10) 
```
## 4 Define Model
```python
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from tensorflow.keras.models import Sequential
```
```python
model = Sequential()
```
## 5 Formation of Model
```python
#BLOCK:1 
model.add(Conv2D(filters=64,kernel_size=(4,4),input_shape=(28,28,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#BLOCK:2
model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=(28,28,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2))) 

model.add(Flatten())
model.add(Dense(256,activation='relu'))

model.add(Dense(10,activation='softmax'))
```
## 6 Compile Model
```python
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
```
## 7 Summary of Model
```python
model.summary() 
```
### Summary:
```python
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_4 (Conv2D)            (None, 25, 25, 64)        1088      
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 12, 12, 64)        0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 9, 9, 32)          32800     
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 4, 4, 32)          0         
_________________________________________________________________
flatten_2 (Flatten)          (None, 512)               0         
_________________________________________________________________
dense_4 (Dense)              (None, 256)               131328    
_________________________________________________________________
dense_5 (Dense)              (None, 10)                2570      
=================================================================
Total params: 167,786
Trainable params: 167,786
Non-trainable params: 0
_________________________________________________________________
```
## 8 Training of model
```python
model.fit(x_train,y_train,verbose=1,epochs=10)  
```
## 9 Evaluation of Model
```python
model.evaluate(x_test,y_test) 
```
## 10 Generation of Classification Report
```python
import sklearn
from sklearn.metrics import classification_report
prediction = model.predict_classes(x_test) 
```
```python
print(classification_report(y_test,prediction)) 
```
### Classificatin Report
```python
precision    recall  f1-score   support

           0       0.79      0.91      0.85      1000
           1       1.00      0.98      0.99      1000
           2       0.76      0.90      0.83      1000
           3       0.90      0.93      0.92      1000
           4       0.92      0.72      0.80      1000
           5       0.98      0.98      0.98      1000
           6       0.77      0.66      0.71      1000
           7       0.96      0.96      0.96      1000
           8       0.96      0.98      0.97      1000
           9       0.97      0.96      0.97      1000

    accuracy                           0.90     10000
   macro avg       0.90      0.90      0.90     10000
weighted avg       0.90      0.90      0.90     10000
```
# Installation:
open command prompt ant type the following command:
```
git clone https://github.com/Mahnoor123-Fatima/Image-Classifier-for-Fashion_MNIST-Dataset.git
```

# Enjoy!

--------------------------------------------------------------------------------------------------------


[![Open Source Love svg1](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](#)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat&label=Contributions&colorA=red&colorB=black	)](#)
[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](#)
