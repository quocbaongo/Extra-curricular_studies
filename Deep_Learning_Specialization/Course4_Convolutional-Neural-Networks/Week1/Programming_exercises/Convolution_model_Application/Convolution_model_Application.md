# Convolutional Neural Networks: Application

Welcome to Course 4's second assignment! In this notebook, you will:

- Create a mood classifer using the TF Keras Sequential API
- Build a ConvNet to identify sign language digits using the TF Keras Functional API

**After this assignment you will be able to:**

- Build and train a ConvNet in TensorFlow for a __binary__ classification problem
- Build and train a ConvNet in TensorFlow for a __multiclass__ classification problem
- Explain different use cases for the Sequential and Functional APIs

To complete this assignment, you should already be familiar with TensorFlow. If you are not, please refer back to the **TensorFlow Tutorial** of the third week of Course 2 ("**Improving deep neural networks**").

## Important Note on Submission to the AutoGrader

Before submitting your assignment to the AutoGrader, please make sure you are not doing the following:

1. You have not added any _extra_ `print` statement(s) in the assignment.
2. You have not added any _extra_ code cell(s) in the assignment.
3. You have not changed any of the function parameters.
4. You are not using any global variables inside your graded exercises. Unless specifically instructed to do so, please refrain from it and use the local variables instead.
5. You are not changing the assignment code where it is not required, like creating _extra_ variables.

If you do any of the following, you will get something like, `Grader Error: Grader feedback not found` (or similarly unexpected) error upon submitting your assignment. Before asking for help/debugging the errors in your assignment, check for these first. If this is the case, and you don't remember the changes you have made, you can get a fresh copy of the assignment by following these [instructions](https://www.coursera.org/learn/convolutional-neural-networks/supplement/DS4yP/h-ow-to-refresh-your-workspace).

## Table of Contents

- [1 - Packages](#1)
    - [1.1 - Load the Data and Split the Data into Train/Test Sets](#1-1)
- [2 - Layers in TF Keras](#2)
- [3 - The Sequential API](#3)
    - [3.1 - Create the Sequential Model](#3-1)
        - [Exercise 1 - happyModel](#ex-1)
    - [3.2 - Train and Evaluate the Model](#3-2)
- [4 - The Functional API](#4)
    - [4.1 - Load the SIGNS Dataset](#4-1)
    - [4.2 - Split the Data into Train/Test Sets](#4-2)
    - [4.3 - Forward Propagation](#4-3)
        - [Exercise 2 - convolutional_model](#ex-2)
    - [4.4 - Train the Model](#4-4)
- [5 - History Object](#5)
- [6 - Bibliography](#6)

<a name='1'></a>
## 1 - Packages

As usual, begin by loading in the packages.


```python
### v1.1
```


```python
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
from PIL import Image
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.python.framework import ops
from cnn_utils import *
from test_utils import summary, comparator

%matplotlib inline
np.random.seed(1)
```

<a name='1-1'></a>
### 1.1 - Load the Data and Split the Data into Train/Test Sets

You'll be using the Happy House dataset for this part of the assignment, which contains images of peoples' faces. Your task will be to build a ConvNet that determines whether the people in the images are smiling or not -- because they only get to enter the house if they're smiling!  


```python
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_happy_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
```

    number of training examples = 600
    number of test examples = 150
    X_train shape: (600, 64, 64, 3)
    Y_train shape: (600, 1)
    X_test shape: (150, 64, 64, 3)
    Y_test shape: (150, 1)


You can display the images contained in the dataset. Images are **64x64** pixels in RGB format (3 channels).


```python
index = 124
plt.imshow(X_train_orig[index]) #display sample training image
plt.show()
```


![png](output_8_0.png)


<a name='2'></a>
## 2 - Layers in TF Keras 

In the previous assignment, you created layers manually in numpy. In TF Keras, you don't have to write code directly to create layers. Rather, TF Keras has pre-defined layers you can use. 

When you create a layer in TF Keras, you are creating a function that takes some input and transforms it into an output you can reuse later. Nice and easy! 

<a name='3'></a>
## 3 - The Sequential API

In the previous assignment, you built helper functions using `numpy` to understand the mechanics behind convolutional neural networks. Most practical applications of deep learning today are built using programming frameworks, which have many built-in functions you can simply call. Keras is a high-level abstraction built on top of TensorFlow, which allows for even more simplified and optimized model creation and training. 

For the first part of this assignment, you'll create a model using TF Keras' Sequential API, which allows you to build layer by layer, and is ideal for building models where each layer has **exactly one** input tensor and **one** output tensor. 

As you'll see, using the Sequential API is simple and straightforward, but is only appropriate for simpler, more straightforward tasks. Later in this notebook you'll spend some time building with a more flexible, powerful alternative: the Functional API. 
 

<a name='3-1'></a>
### 3.1 - Create the Sequential Model

As mentioned earlier, the TensorFlow Keras Sequential API can be used to build simple models with layer operations that proceed in a sequential order. 

You can also add layers incrementally to a Sequential model with the `.add()` method, or remove them using the `.pop()` method, much like you would in a regular Python list.

Actually, you can think of a Sequential model as behaving like a list of layers. Like Python lists, Sequential layers are ordered, and the order in which they are specified matters.  If your model is non-linear or contains layers with multiple inputs or outputs, a Sequential model wouldn't be the right choice!

For any layer construction in Keras, you'll need to specify the input shape in advance. This is because in Keras, the shape of the weights is based on the shape of the inputs. The weights are only created when the model first sees some input data. Sequential models can be created by passing a list of layers to the Sequential constructor, like you will do in the next assignment.

<a name='ex-1'></a>
### Exercise 1 - happyModel

Implement the `happyModel` function below to build the following model: `ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE`. Take help from [tf.keras.layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers) 

Also, plug in the following parameters for all the steps:

 - [ZeroPadding2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding2D): padding 3, input shape 64 x 64 x 3
 - [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D): Use 32 7x7 filters, stride 1
 - [BatchNormalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization): for axis 3
 - [ReLU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU)
 - [MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D): Using default parameters
 - [Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten) the previous output.
 - Fully-connected ([Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)) layer: Apply a fully connected layer with 1 neuron and a sigmoid activation. 
 
 
 **Hint:**
 
 Use **tfl** as shorthand for **tensorflow.keras.layers**


```python
# GRADED FUNCTION: happyModel

def happyModel():
    """
    Implements the forward propagation for the binary classification model:
    ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Note that for simplicity and grading purposes, you'll hard-code all the values
    such as the stride and kernel (filter) sizes. 
    Normally, functions should take these values as function parameters.
    
    Arguments:
    None

    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """
    model = tf.keras.Sequential([
            ## ZeroPadding2D with padding 3, input shape of 64 x 64 x 3
            
            ## Conv2D with 32 7x7 filters and stride of 1
            
            ## BatchNormalization for axis 3
            
            ## ReLU
            
            ## Max Pooling 2D with default parameters
            
            ## Flatten layer
            
            ## Dense layer with 1 unit for output & 'sigmoid' activation
            
            # YOUR CODE STARTS HERE
            tf.keras.Input(shape=(64, 64, 3)),
            tfl.ZeroPadding2D(padding=3),
            tfl.Conv2D(32, (7, 7), strides=1),
            tfl.BatchNormalization(axis=3),
            tfl.ReLU(),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dense(1, activation='sigmoid')
            # YOUR CODE ENDS HERE
        ])
    
    return model
```


```python
happy_model = happyModel()
# Print a summary for each layer
for layer in summary(happy_model):
    print(layer)
    
output = [['ZeroPadding2D', (None, 70, 70, 3), 0, ((3, 3), (3, 3))],
            ['Conv2D', (None, 64, 64, 32), 4736, 'valid', 'linear', 'GlorotUniform'],
            ['BatchNormalization', (None, 64, 64, 32), 128],
            ['ReLU', (None, 64, 64, 32), 0],
            ['MaxPooling2D', (None, 32, 32, 32), 0, (2, 2), (2, 2), 'valid'],
            ['Flatten', (None, 32768), 0],
            ['Dense', (None, 1), 32769, 'sigmoid']]
    
comparator(summary(happy_model), output)
```

    ['ZeroPadding2D', (None, 70, 70, 3), 0, ((3, 3), (3, 3))]
    ['Conv2D', (None, 64, 64, 32), 4736, 'valid', 'linear', 'GlorotUniform']
    ['BatchNormalization', (None, 64, 64, 32), 128]
    ['ReLU', (None, 64, 64, 32), 0]
    ['MaxPooling2D', (None, 32, 32, 32), 0, (2, 2), (2, 2), 'valid']
    ['Flatten', (None, 32768), 0]
    ['Dense', (None, 1), 32769, 'sigmoid']
    [32mAll tests passed![0m


#### Expected Output:

```
['ZeroPadding2D', (None, 70, 70, 3), 0, ((3, 3), (3, 3))]
['Conv2D', (None, 64, 64, 32), 4736, 'valid', 'linear', 'GlorotUniform']
['BatchNormalization', (None, 64, 64, 32), 128]
['ReLU', (None, 64, 64, 32), 0]
['MaxPooling2D', (None, 32, 32, 32), 0, (2, 2), (2, 2), 'valid']
['Flatten', (None, 32768), 0]
['Dense', (None, 1), 32769, 'sigmoid']
All tests passed!
```

Now that your model is created, you can compile it for training with an optimizer and loss of your choice. When the string `accuracy` is specified as a metric, the type of accuracy used will be automatically converted based on the loss function used. This is one of the many optimizations built into TensorFlow that make your life easier! If you'd like to read more on how the compiler operates, check the docs [here](https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile).


```python
happy_model.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
```

It's time to check your model's parameters with the `.summary()` method. This will display the types of layers you have, the shape of the outputs, and how many parameters are in each layer. 


```python
happy_model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    zero_padding2d_1 (ZeroPaddin (None, 70, 70, 3)         0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 64, 64, 32)        4736      
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 64, 64, 32)        128       
    _________________________________________________________________
    re_lu_1 (ReLU)               (None, 64, 64, 32)        0         
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 32, 32, 32)        0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 32768)             0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 1)                 32769     
    =================================================================
    Total params: 37,633
    Trainable params: 37,569
    Non-trainable params: 64
    _________________________________________________________________


<a name='3-2'></a>
### 3.2 - Train and Evaluate the Model

After creating the model, compiling it with your choice of optimizer and loss function, and doing a sanity check on its contents, you are now ready to build! 

Simply call `.fit()` to train. That's it! No need for mini-batching, saving, or complex backpropagation computations. That's all been done for you, as you're using a TensorFlow dataset with the batches specified already. You do have the option to specify epoch number or minibatch size if you like (for example, in the case of an un-batched dataset).


```python
happy_model.fit(X_train, Y_train, epochs=10, batch_size=16)
```

    Epoch 1/10
    38/38 [==============================] - 4s 100ms/step - loss: 0.7875 - accuracy: 0.7850
    Epoch 2/10
    38/38 [==============================] - 4s 100ms/step - loss: 0.3545 - accuracy: 0.8633
    Epoch 3/10
    38/38 [==============================] - 4s 97ms/step - loss: 0.1802 - accuracy: 0.9367
    Epoch 4/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.1247 - accuracy: 0.9483
    Epoch 5/10
    38/38 [==============================] - 4s 97ms/step - loss: 0.1402 - accuracy: 0.9483 1s - los
    Epoch 6/10
    38/38 [==============================] - 4s 97ms/step - loss: 0.1016 - accuracy: 0.9617
    Epoch 7/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.0993 - accuracy: 0.9633
    Epoch 8/10
    38/38 [==============================] - 4s 97ms/step - loss: 0.0859 - accuracy: 0.9717
    Epoch 9/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.0754 - accuracy: 0.9750
    Epoch 10/10
    38/38 [==============================] - 4s 97ms/step - loss: 0.0675 - accuracy: 0.9800





    <tensorflow.python.keras.callbacks.History at 0x768c8f7d6b90>



After that completes, just use `.evaluate()` to evaluate against your test set. This function will print the value of the loss function and the performance metrics specified during the compilation of the model. In this case, the `binary_crossentropy` and the `accuracy` respectively.


```python
happy_model.evaluate(X_test, Y_test)
```

    5/5 [==============================] - 0s 25ms/step - loss: 0.1323 - accuracy: 0.9467





    [0.13227106630802155, 0.9466666579246521]



Easy, right? But what if you need to build a model with shared layers, branches, or multiple inputs and outputs? This is where Sequential, with its beautifully simple yet limited functionality, won't be able to help you. 

Next up: Enter the Functional API, your slightly more complex, highly flexible friend.  

<a name='4'></a>
## 4 - The Functional API

Welcome to the second half of the assignment, where you'll use Keras' flexible [Functional API](https://www.tensorflow.org/guide/keras/functional) to build a ConvNet that can differentiate between 6 sign language digits. 

The Functional API can handle models with non-linear topology, shared layers, as well as layers with multiple inputs or outputs. Imagine that, where the Sequential API requires the model to move in a linear fashion through its layers, the Functional API allows much more flexibility. Where Sequential is a straight line, a Functional model is a graph, where the nodes of the layers can connect in many more ways than one. 

In the visual example below, the one possible direction of the movement Sequential model is shown in contrast to a skip connection, which is just one of the many ways a Functional model can be constructed. A skip connection, as you might have guessed, skips some layer in the network and feeds the output to a later layer in the network. Don't worry, you'll be spending more time with skip connections very soon! 

<img src="images/seq_vs_func.png" style="width:350px;height:200px;">

<a name='4-1'></a>
### 4.1 - Load the SIGNS Dataset

As a reminder, the SIGNS dataset is a collection of 6 signs representing numbers from 0 to 5.


```python
# Loading the data (signs)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_signs_dataset()
```

<img src="images/SIGNS.png" style="width:800px;height:300px;">

The next cell will show you an example of a labelled image in the dataset. Feel free to change the value of `index` below and re-run to see different examples. 


```python
# Example of an image from the dataset
index = 9
plt.imshow(X_train_orig[index])
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))
```

    y = 4



![png](output_30_1.png)


<a name='4-2'></a>
### 4.2 - Split the Data into Train/Test Sets

In Course 2, you built a fully-connected network for this dataset. But since this is an image dataset, it is more natural to apply a ConvNet to it.

To get started, let's examine the shapes of your data. 


```python
X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
```

    number of training examples = 1080
    number of test examples = 120
    X_train shape: (1080, 64, 64, 3)
    Y_train shape: (1080, 6)
    X_test shape: (120, 64, 64, 3)
    Y_test shape: (120, 6)


<a name='4-3'></a>
### 4.3 - Forward Propagation

In TensorFlow, there are built-in functions that implement the convolution steps for you. By now, you should be familiar with how TensorFlow builds computational graphs. In the [Functional API](https://www.tensorflow.org/guide/keras/functional), you create a graph of layers. This is what allows such great flexibility.

However, the following model could also be defined using the Sequential API since the information flow is on a single line. But don't deviate. What we want you to learn is to use the functional API.

Begin building your graph of layers by creating an input node that functions as a callable object:

- **input_img = tf.keras.Input(shape=input_shape):** 

Then, create a new node in the graph of layers by calling a layer on the `input_img` object: 

- **tf.keras.layers.Conv2D(filters= ... , kernel_size= ... , padding='same')(input_img):** Read the full documentation on [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D).

- **tf.keras.layers.MaxPool2D(pool_size=(f, f), strides=(s, s), padding='same'):** `MaxPool2D()` downsamples your input using a window of size (f, f) and strides of size (s, s) to carry out max pooling over each window.  For max pooling, you usually operate on a single example at a time and a single channel at a time. Read the full documentation on [MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D).

- **tf.keras.layers.ReLU():** computes the elementwise ReLU of Z (which can be any shape). You can read the full documentation on [ReLU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU).

- **tf.keras.layers.Flatten()**: given a tensor "P", this function takes each training (or test) example in the batch and flattens it into a 1D vector.  

    * If a tensor P has the shape (batch_size,h,w,c), it returns a flattened tensor with shape (batch_size, k), where $k=h \times w \times c$.  "k" equals the product of all the dimension sizes other than the first dimension.
    
    * For example, given a tensor with dimensions [100, 2, 3, 4], it flattens the tensor to be of shape [100, 24], where 24 = 2 * 3 * 4.  You can read the full documentation on [Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten).

- **tf.keras.layers.Dense(units= ... , activation='softmax')(F):** given the flattened input F, it returns the output computed using a fully connected layer. You can read the full documentation on [Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense).

In the last function above (`tf.keras.layers.Dense()`), the fully connected layer automatically initializes weights in the graph and keeps on training them as you train the model. Hence, you did not need to initialize those weights when initializing the parameters.

Lastly, before creating the model, you'll need to define the output using the last of the function's compositions (in this example, a Dense layer): 

- **outputs = tf.keras.layers.Dense(units=6, activation='softmax')(F)**


#### Window, kernel, filter, pool

The words "kernel" and "filter" are used to refer to the same thing. The word "filter" accounts for the amount of "kernels" that will be used in a single convolution layer. "Pool" is the name of the operation that takes the max or average value of the kernels. 

This is why the parameter `pool_size` refers to `kernel_size`, and you use `(f,f)` to refer to the filter size. 

Pool size and kernel size refer to the same thing in different objects - They refer to the shape of the window where the operation takes place. 

<a name='ex-2'></a>
### Exercise 2 - convolutional_model

Implement the `convolutional_model` function below to build the following model: `CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE`. Use the functions above! 

Also, plug in the following parameters for all the steps:

 - [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D): Use 8 4 by 4 filters, stride 1, padding is "SAME"
 - [ReLU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU)
 - [MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D): Use an 8 by 8 filter size and an 8 by 8 stride, padding is "SAME"
 - **Conv2D**: Use 16 2 by 2 filters, stride 1, padding is "SAME"
 - **ReLU**
 - **MaxPool2D**: Use a 4 by 4 filter size and a 4 by 4 stride, padding is "SAME"
 - [Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten) the previous output.
 - Fully-connected ([Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)) layer: Apply a fully connected layer with 6 neurons and a softmax activation. 


```python
# GRADED FUNCTION: convolutional_model

def convolutional_model(input_shape):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Note that for simplicity and grading purposes, you'll hard-code some values
    such as the stride and kernel (filter) sizes. 
    Normally, functions should take these values as function parameters.
    
    Arguments:
    input_img -- input dataset, of shape (input_shape)

    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """

    input_img = tf.keras.Input(shape=input_shape)
    ## CONV2D: 8 filters 4x4, stride of 1, padding 'SAME'
    # Z1 = None
    ## RELU
    # A1 = None
    ## MAXPOOL: window 8x8, stride 8, padding 'SAME'
    # P1 = None
    ## CONV2D: 16 filters 2x2, stride 1, padding 'SAME'
    # Z2 = None
    ## RELU
    # A2 = None
    ## MAXPOOL: window 4x4, stride 4, padding 'SAME'
    # P2 = None
    ## FLATTEN
    # F = None
    ## Dense layer
    ## 6 neurons in output layer. Hint: one of the arguments should be "activation='softmax'" 
    # outputs = None
    # YOUR CODE STARTS HERE
    Z1 = tfl.Conv2D(8, (4, 4), strides=1, padding="same")(input_img)
    A1 = tfl.ReLU()(Z1)
    P1 = tfl.MaxPool2D(pool_size=(8, 8), strides=8, padding='same')(A1)
    
    Z2 = tfl.Conv2D(16, (2, 2), strides=1, padding="same")(P1)
    A2 = tfl.ReLU()(Z2)
    P2 = tfl.MaxPool2D(pool_size=(4, 4), strides=4, padding='same')(A2)
    
    F = tfl.Flatten()(P2)
    outputs = tfl.Dense(6, activation='softmax')(F)
    # YOUR CODE ENDS HERE
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model
```


```python
conv_model = convolutional_model((64, 64, 3))
conv_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
conv_model.summary()
    
output = [['InputLayer', [(None, 64, 64, 3)], 0],
        ['Conv2D', (None, 64, 64, 8), 392, 'same', 'linear', 'GlorotUniform'],
        ['ReLU', (None, 64, 64, 8), 0],
        ['MaxPooling2D', (None, 8, 8, 8), 0, (8, 8), (8, 8), 'same'],
        ['Conv2D', (None, 8, 8, 16), 528, 'same', 'linear', 'GlorotUniform'],
        ['ReLU', (None, 8, 8, 16), 0],
        ['MaxPooling2D', (None, 2, 2, 16), 0, (4, 4), (4, 4), 'same'],
        ['Flatten', (None, 64), 0],
        ['Dense', (None, 6), 390, 'softmax']]
    
comparator(summary(conv_model), output)
```

    Model: "functional_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_4 (InputLayer)         [(None, 64, 64, 3)]       0         
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 64, 64, 8)         392       
    _________________________________________________________________
    re_lu_4 (ReLU)               (None, 64, 64, 8)         0         
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 8, 8, 8)           0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 8, 8, 16)          528       
    _________________________________________________________________
    re_lu_5 (ReLU)               (None, 8, 8, 16)          0         
    _________________________________________________________________
    max_pooling2d_5 (MaxPooling2 (None, 2, 2, 16)          0         
    _________________________________________________________________
    flatten_3 (Flatten)          (None, 64)                0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 6)                 390       
    =================================================================
    Total params: 1,310
    Trainable params: 1,310
    Non-trainable params: 0
    _________________________________________________________________
    [32mAll tests passed![0m


Both the Sequential and Functional APIs return a TF Keras model object. The only difference is how inputs are handled inside the object model! 

<a name='4-4'></a>
### 4.4 - Train the Model


```python
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(64)
history = conv_model.fit(train_dataset, epochs=100, validation_data=test_dataset)
```

    Epoch 1/100
    17/17 [==============================] - 2s 102ms/step - loss: 1.8300 - accuracy: 0.1667 - val_loss: 1.7936 - val_accuracy: 0.1667
    Epoch 2/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.7931 - accuracy: 0.1694 - val_loss: 1.7831 - val_accuracy: 0.2000
    Epoch 3/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.7872 - accuracy: 0.1880 - val_loss: 1.7799 - val_accuracy: 0.2500
    Epoch 4/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.7836 - accuracy: 0.2194 - val_loss: 1.7762 - val_accuracy: 0.3000
    Epoch 5/100
    17/17 [==============================] - 2s 95ms/step - loss: 1.7792 - accuracy: 0.2315 - val_loss: 1.7684 - val_accuracy: 0.3000
    Epoch 6/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.7733 - accuracy: 0.2250 - val_loss: 1.7611 - val_accuracy: 0.3000
    Epoch 7/100
    17/17 [==============================] - 2s 95ms/step - loss: 1.7663 - accuracy: 0.2759 - val_loss: 1.7522 - val_accuracy: 0.3583
    Epoch 8/100
    17/17 [==============================] - 2s 96ms/step - loss: 1.7580 - accuracy: 0.3046 - val_loss: 1.7424 - val_accuracy: 0.3750
    Epoch 9/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.7480 - accuracy: 0.3296 - val_loss: 1.7308 - val_accuracy: 0.4083
    Epoch 10/100
    17/17 [==============================] - 2s 99ms/step - loss: 1.7358 - accuracy: 0.3648 - val_loss: 1.7157 - val_accuracy: 0.4167
    Epoch 11/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.7208 - accuracy: 0.3917 - val_loss: 1.6986 - val_accuracy: 0.5167
    Epoch 12/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.7017 - accuracy: 0.4537 - val_loss: 1.6784 - val_accuracy: 0.5667
    Epoch 13/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.6785 - accuracy: 0.4954 - val_loss: 1.6542 - val_accuracy: 0.5500
    Epoch 14/100
    17/17 [==============================] - 2s 95ms/step - loss: 1.6505 - accuracy: 0.5259 - val_loss: 1.6259 - val_accuracy: 0.6000
    Epoch 15/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.6173 - accuracy: 0.5444 - val_loss: 1.5923 - val_accuracy: 0.5833
    Epoch 16/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.5780 - accuracy: 0.5537 - val_loss: 1.5549 - val_accuracy: 0.6250
    Epoch 17/100
    17/17 [==============================] - 2s 101ms/step - loss: 1.5327 - accuracy: 0.5676 - val_loss: 1.5117 - val_accuracy: 0.6000
    Epoch 18/100
    17/17 [==============================] - 2s 96ms/step - loss: 1.4812 - accuracy: 0.5750 - val_loss: 1.4632 - val_accuracy: 0.6250
    Epoch 19/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.4245 - accuracy: 0.6056 - val_loss: 1.4080 - val_accuracy: 0.5917
    Epoch 20/100
    17/17 [==============================] - 2s 101ms/step - loss: 1.3690 - accuracy: 0.6102 - val_loss: 1.3544 - val_accuracy: 0.6000
    Epoch 21/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.3140 - accuracy: 0.6231 - val_loss: 1.3022 - val_accuracy: 0.6000
    Epoch 22/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.2627 - accuracy: 0.6333 - val_loss: 1.2507 - val_accuracy: 0.6167
    Epoch 23/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.2109 - accuracy: 0.6546 - val_loss: 1.2021 - val_accuracy: 0.6667
    Epoch 24/100
    17/17 [==============================] - 2s 95ms/step - loss: 1.1609 - accuracy: 0.6713 - val_loss: 1.1539 - val_accuracy: 0.6833
    Epoch 25/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.1144 - accuracy: 0.6750 - val_loss: 1.1144 - val_accuracy: 0.7000
    Epoch 26/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.0731 - accuracy: 0.6806 - val_loss: 1.0744 - val_accuracy: 0.7083
    Epoch 27/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.0338 - accuracy: 0.6907 - val_loss: 1.0384 - val_accuracy: 0.7167
    Epoch 28/100
    17/17 [==============================] - 2s 95ms/step - loss: 0.9968 - accuracy: 0.7130 - val_loss: 1.0069 - val_accuracy: 0.7083
    Epoch 29/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.9661 - accuracy: 0.7213 - val_loss: 0.9783 - val_accuracy: 0.7167
    Epoch 30/100
    17/17 [==============================] - 2s 99ms/step - loss: 0.9349 - accuracy: 0.7269 - val_loss: 0.9488 - val_accuracy: 0.7167
    Epoch 31/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.9087 - accuracy: 0.7324 - val_loss: 0.9296 - val_accuracy: 0.7167
    Epoch 32/100
    17/17 [==============================] - 2s 95ms/step - loss: 0.8815 - accuracy: 0.7398 - val_loss: 0.9021 - val_accuracy: 0.7417
    Epoch 33/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.8604 - accuracy: 0.7481 - val_loss: 0.8832 - val_accuracy: 0.7333
    Epoch 34/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.8380 - accuracy: 0.7565 - val_loss: 0.8641 - val_accuracy: 0.7417
    Epoch 35/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.8179 - accuracy: 0.7639 - val_loss: 0.8457 - val_accuracy: 0.7667
    Epoch 36/100
    17/17 [==============================] - 2s 95ms/step - loss: 0.7989 - accuracy: 0.7685 - val_loss: 0.8321 - val_accuracy: 0.7500
    Epoch 37/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.7811 - accuracy: 0.7750 - val_loss: 0.8139 - val_accuracy: 0.7500
    Epoch 38/100
    17/17 [==============================] - 2s 96ms/step - loss: 0.7649 - accuracy: 0.7722 - val_loss: 0.8008 - val_accuracy: 0.7583
    Epoch 39/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.7506 - accuracy: 0.7806 - val_loss: 0.7897 - val_accuracy: 0.7750
    Epoch 40/100
    17/17 [==============================] - 2s 105ms/step - loss: 0.7344 - accuracy: 0.7815 - val_loss: 0.7745 - val_accuracy: 0.7667
    Epoch 41/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.7219 - accuracy: 0.7898 - val_loss: 0.7642 - val_accuracy: 0.7833
    Epoch 42/100
    17/17 [==============================] - 2s 96ms/step - loss: 0.7074 - accuracy: 0.7907 - val_loss: 0.7515 - val_accuracy: 0.7833
    Epoch 43/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.6951 - accuracy: 0.7944 - val_loss: 0.7421 - val_accuracy: 0.7833
    Epoch 44/100
    17/17 [==============================] - 2s 95ms/step - loss: 0.6824 - accuracy: 0.8000 - val_loss: 0.7313 - val_accuracy: 0.7750
    Epoch 45/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.6714 - accuracy: 0.8037 - val_loss: 0.7220 - val_accuracy: 0.7750
    Epoch 46/100
    17/17 [==============================] - 2s 95ms/step - loss: 0.6598 - accuracy: 0.8056 - val_loss: 0.7118 - val_accuracy: 0.7833
    Epoch 47/100
    17/17 [==============================] - 2s 96ms/step - loss: 0.6495 - accuracy: 0.8074 - val_loss: 0.7032 - val_accuracy: 0.7833
    Epoch 48/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.6392 - accuracy: 0.8083 - val_loss: 0.6948 - val_accuracy: 0.7917
    Epoch 49/100
    17/17 [==============================] - 2s 95ms/step - loss: 0.6296 - accuracy: 0.8093 - val_loss: 0.6870 - val_accuracy: 0.7917
    Epoch 50/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.6200 - accuracy: 0.8120 - val_loss: 0.6792 - val_accuracy: 0.8000
    Epoch 51/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.6111 - accuracy: 0.8130 - val_loss: 0.6713 - val_accuracy: 0.8000
    Epoch 52/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.6025 - accuracy: 0.8148 - val_loss: 0.6641 - val_accuracy: 0.8000
    Epoch 53/100
    17/17 [==============================] - 2s 101ms/step - loss: 0.5942 - accuracy: 0.8213 - val_loss: 0.6563 - val_accuracy: 0.8083
    Epoch 54/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.5858 - accuracy: 0.8194 - val_loss: 0.6488 - val_accuracy: 0.8083
    Epoch 55/100
    17/17 [==============================] - 2s 101ms/step - loss: 0.5779 - accuracy: 0.8222 - val_loss: 0.6418 - val_accuracy: 0.8083
    Epoch 56/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.5701 - accuracy: 0.8250 - val_loss: 0.6348 - val_accuracy: 0.8083
    Epoch 57/100
    17/17 [==============================] - 2s 95ms/step - loss: 0.5628 - accuracy: 0.8287 - val_loss: 0.6285 - val_accuracy: 0.8083
    Epoch 58/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.5556 - accuracy: 0.8333 - val_loss: 0.6215 - val_accuracy: 0.8083
    Epoch 59/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.5483 - accuracy: 0.8370 - val_loss: 0.6157 - val_accuracy: 0.8083
    Epoch 60/100
    17/17 [==============================] - 2s 96ms/step - loss: 0.5414 - accuracy: 0.8398 - val_loss: 0.6103 - val_accuracy: 0.8083
    Epoch 61/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.5348 - accuracy: 0.8426 - val_loss: 0.6051 - val_accuracy: 0.8083
    Epoch 62/100
    17/17 [==============================] - 2s 95ms/step - loss: 0.5285 - accuracy: 0.8463 - val_loss: 0.5997 - val_accuracy: 0.8083
    Epoch 63/100
    17/17 [==============================] - 2s 96ms/step - loss: 0.5221 - accuracy: 0.8454 - val_loss: 0.5947 - val_accuracy: 0.8083
    Epoch 64/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.5156 - accuracy: 0.8500 - val_loss: 0.5900 - val_accuracy: 0.7917
    Epoch 65/100
    17/17 [==============================] - 2s 101ms/step - loss: 0.5094 - accuracy: 0.8509 - val_loss: 0.5858 - val_accuracy: 0.7917
    Epoch 66/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.5037 - accuracy: 0.8537 - val_loss: 0.5816 - val_accuracy: 0.7917
    Epoch 67/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.4985 - accuracy: 0.8537 - val_loss: 0.5768 - val_accuracy: 0.7917
    Epoch 68/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.4927 - accuracy: 0.8565 - val_loss: 0.5730 - val_accuracy: 0.7917
    Epoch 69/100
    17/17 [==============================] - 2s 97ms/step - loss: 0.4872 - accuracy: 0.8574 - val_loss: 0.5691 - val_accuracy: 0.7917
    Epoch 70/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.4818 - accuracy: 0.8583 - val_loss: 0.5652 - val_accuracy: 0.7917
    Epoch 71/100
    17/17 [==============================] - 2s 96ms/step - loss: 0.4764 - accuracy: 0.8620 - val_loss: 0.5617 - val_accuracy: 0.8000
    Epoch 72/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.4714 - accuracy: 0.8620 - val_loss: 0.5583 - val_accuracy: 0.8083
    Epoch 73/100
    17/17 [==============================] - 2s 101ms/step - loss: 0.4664 - accuracy: 0.8685 - val_loss: 0.5549 - val_accuracy: 0.8083
    Epoch 74/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.4615 - accuracy: 0.8685 - val_loss: 0.5529 - val_accuracy: 0.8083
    Epoch 75/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.4570 - accuracy: 0.8704 - val_loss: 0.5493 - val_accuracy: 0.8083
    Epoch 76/100
    17/17 [==============================] - 2s 96ms/step - loss: 0.4522 - accuracy: 0.8741 - val_loss: 0.5463 - val_accuracy: 0.8167
    Epoch 77/100
    17/17 [==============================] - 2s 96ms/step - loss: 0.4476 - accuracy: 0.8750 - val_loss: 0.5436 - val_accuracy: 0.8167
    Epoch 78/100
    17/17 [==============================] - 2s 102ms/step - loss: 0.4434 - accuracy: 0.8787 - val_loss: 0.5409 - val_accuracy: 0.8167
    Epoch 79/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.4389 - accuracy: 0.8815 - val_loss: 0.5381 - val_accuracy: 0.8167
    Epoch 80/100
    17/17 [==============================] - 2s 96ms/step - loss: 0.4348 - accuracy: 0.8815 - val_loss: 0.5354 - val_accuracy: 0.8167
    Epoch 81/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.4307 - accuracy: 0.8815 - val_loss: 0.5329 - val_accuracy: 0.8167
    Epoch 82/100
    17/17 [==============================] - 2s 101ms/step - loss: 0.4267 - accuracy: 0.8824 - val_loss: 0.5303 - val_accuracy: 0.8167
    Epoch 83/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.4229 - accuracy: 0.8843 - val_loss: 0.5280 - val_accuracy: 0.8167
    Epoch 84/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.4193 - accuracy: 0.8870 - val_loss: 0.5260 - val_accuracy: 0.8167
    Epoch 85/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.4153 - accuracy: 0.8861 - val_loss: 0.5234 - val_accuracy: 0.8167
    Epoch 86/100
    17/17 [==============================] - 2s 95ms/step - loss: 0.4116 - accuracy: 0.8889 - val_loss: 0.5206 - val_accuracy: 0.8167
    Epoch 87/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.4080 - accuracy: 0.8898 - val_loss: 0.5190 - val_accuracy: 0.8167
    Epoch 88/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.4042 - accuracy: 0.8898 - val_loss: 0.5163 - val_accuracy: 0.8167
    Epoch 89/100
    17/17 [==============================] - 2s 95ms/step - loss: 0.4005 - accuracy: 0.8898 - val_loss: 0.5138 - val_accuracy: 0.8167
    Epoch 90/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.3971 - accuracy: 0.8898 - val_loss: 0.5119 - val_accuracy: 0.8167
    Epoch 91/100
    17/17 [==============================] - 2s 95ms/step - loss: 0.3936 - accuracy: 0.8889 - val_loss: 0.5098 - val_accuracy: 0.8167
    Epoch 92/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.3903 - accuracy: 0.8889 - val_loss: 0.5076 - val_accuracy: 0.8167
    Epoch 93/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.3870 - accuracy: 0.8917 - val_loss: 0.5061 - val_accuracy: 0.8167
    Epoch 94/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.3837 - accuracy: 0.8926 - val_loss: 0.5042 - val_accuracy: 0.8167
    Epoch 95/100
    17/17 [==============================] - 2s 95ms/step - loss: 0.3806 - accuracy: 0.8926 - val_loss: 0.5027 - val_accuracy: 0.8167
    Epoch 96/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.3777 - accuracy: 0.8935 - val_loss: 0.5011 - val_accuracy: 0.8167
    Epoch 97/100
    17/17 [==============================] - 2s 96ms/step - loss: 0.3748 - accuracy: 0.8944 - val_loss: 0.5006 - val_accuracy: 0.8167
    Epoch 98/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.3720 - accuracy: 0.8963 - val_loss: 0.4992 - val_accuracy: 0.8167
    Epoch 99/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.3688 - accuracy: 0.8963 - val_loss: 0.4974 - val_accuracy: 0.8167
    Epoch 100/100
    17/17 [==============================] - 2s 96ms/step - loss: 0.3663 - accuracy: 0.8963 - val_loss: 0.4960 - val_accuracy: 0.8167


<a name='5'></a>
## 5 - History Object 

The history object is an output of the `.fit()` operation, and provides a record of all the loss and metric values in memory. It's stored as a dictionary that you can retrieve at `history.history`: 


```python
history.history
```




    {'loss': [1.8299683332443237,
      1.7930597066879272,
      1.7871549129486084,
      1.7836275100708008,
      1.7792110443115234,
      1.7733160257339478,
      1.766327977180481,
      1.7580033540725708,
      1.7479572296142578,
      1.7358070611953735,
      1.7208459377288818,
      1.7017008066177368,
      1.6785051822662354,
      1.6504859924316406,
      1.6172926425933838,
      1.5780493021011353,
      1.5327045917510986,
      1.4812278747558594,
      1.4245495796203613,
      1.3689634799957275,
      1.3139595985412598,
      1.262681007385254,
      1.2108545303344727,
      1.1608949899673462,
      1.1143624782562256,
      1.0731208324432373,
      1.0337516069412231,
      0.9968242645263672,
      0.9660764336585999,
      0.9349427223205566,
      0.908747673034668,
      0.8815345764160156,
      0.860359251499176,
      0.8379893898963928,
      0.8179206848144531,
      0.7988954782485962,
      0.7811235189437866,
      0.7649406790733337,
      0.7505534887313843,
      0.7344145774841309,
      0.7219451069831848,
      0.7073656320571899,
      0.6950904726982117,
      0.6823576092720032,
      0.6714386343955994,
      0.6597860455513,
      0.6494511365890503,
      0.6392460465431213,
      0.6295955181121826,
      0.6200490593910217,
      0.6111096739768982,
      0.6024848222732544,
      0.5941840410232544,
      0.5857523083686829,
      0.5779175162315369,
      0.5700793862342834,
      0.5628159046173096,
      0.555589497089386,
      0.5483190417289734,
      0.5413647294044495,
      0.5347855091094971,
      0.5284764766693115,
      0.5221098065376282,
      0.5156313180923462,
      0.509430468082428,
      0.5036513805389404,
      0.49854138493537903,
      0.4927261769771576,
      0.48723167181015015,
      0.4817618429660797,
      0.4764039218425751,
      0.4713960289955139,
      0.4663505554199219,
      0.4614962935447693,
      0.45697855949401855,
      0.4522154927253723,
      0.44756338000297546,
      0.44341564178466797,
      0.43894070386886597,
      0.43476924300193787,
      0.43069133162498474,
      0.42669251561164856,
      0.4228622317314148,
      0.41925403475761414,
      0.41530174016952515,
      0.41155117750167847,
      0.407974511384964,
      0.404220312833786,
      0.4004761874675751,
      0.3971099555492401,
      0.3935815393924713,
      0.39026838541030884,
      0.38703224062919617,
      0.383747935295105,
      0.38057833909988403,
      0.3776860535144806,
      0.37484675645828247,
      0.3720126748085022,
      0.3688275218009949,
      0.3662952184677124],
     'accuracy': [0.1666666716337204,
      0.16944444179534912,
      0.18796296417713165,
      0.21944443881511688,
      0.23148147761821747,
      0.22499999403953552,
      0.2759259343147278,
      0.3046296238899231,
      0.3296296298503876,
      0.364814817905426,
      0.3916666805744171,
      0.45370370149612427,
      0.49537035822868347,
      0.5259259343147278,
      0.5444444417953491,
      0.5537037253379822,
      0.5675926208496094,
      0.574999988079071,
      0.605555534362793,
      0.6101852059364319,
      0.6231481432914734,
      0.6333333253860474,
      0.654629647731781,
      0.6712962985038757,
      0.675000011920929,
      0.6805555820465088,
      0.6907407641410828,
      0.7129629850387573,
      0.7212963104248047,
      0.7268518805503845,
      0.7324073910713196,
      0.739814817905426,
      0.7481481432914734,
      0.7564814686775208,
      0.7638888955116272,
      0.7685185074806213,
      0.7749999761581421,
      0.7722222208976746,
      0.7805555462837219,
      0.7814815044403076,
      0.789814829826355,
      0.7907407283782959,
      0.7944444417953491,
      0.800000011920929,
      0.8037037253379822,
      0.8055555820465088,
      0.8074073791503906,
      0.8083333373069763,
      0.8092592358589172,
      0.8120370507240295,
      0.8129629492759705,
      0.8148148059844971,
      0.8212962746620178,
      0.8194444179534912,
      0.8222222328186035,
      0.824999988079071,
      0.8287037014961243,
      0.8333333134651184,
      0.8370370268821716,
      0.8398148417472839,
      0.8425925970077515,
      0.8462963104248047,
      0.845370352268219,
      0.8500000238418579,
      0.8509259223937988,
      0.8537036776542664,
      0.8537036776542664,
      0.8564814925193787,
      0.8574073910713196,
      0.8583333492279053,
      0.8620370626449585,
      0.8620370626449585,
      0.8685185313224792,
      0.8685185313224792,
      0.8703703880310059,
      0.8740741014480591,
      0.875,
      0.8787037134170532,
      0.8814814686775208,
      0.8814814686775208,
      0.8814814686775208,
      0.8824074268341064,
      0.8842592835426331,
      0.8870370388031006,
      0.8861111402511597,
      0.8888888955116272,
      0.8898147940635681,
      0.8898147940635681,
      0.8898147940635681,
      0.8898147940635681,
      0.8888888955116272,
      0.8888888955116272,
      0.8916666507720947,
      0.8925926089286804,
      0.8925926089286804,
      0.8935185074806213,
      0.894444465637207,
      0.8962963223457336,
      0.8962963223457336,
      0.8962963223457336],
     'val_loss': [1.7936047315597534,
      1.7830805778503418,
      1.779883861541748,
      1.7761921882629395,
      1.7684460878372192,
      1.7610958814620972,
      1.7521880865097046,
      1.7424384355545044,
      1.7307509183883667,
      1.7156959772109985,
      1.6985646486282349,
      1.6784411668777466,
      1.6541990041732788,
      1.625938892364502,
      1.592326283454895,
      1.5548596382141113,
      1.5117146968841553,
      1.4632338285446167,
      1.4079910516738892,
      1.3543790578842163,
      1.3021595478057861,
      1.2507402896881104,
      1.202109694480896,
      1.1539382934570312,
      1.1144030094146729,
      1.0743906497955322,
      1.0384174585342407,
      1.0068620443344116,
      0.9783440232276917,
      0.948840856552124,
      0.929572343826294,
      0.902127206325531,
      0.8831838965415955,
      0.8640965819358826,
      0.8456503748893738,
      0.8320521712303162,
      0.8139057159423828,
      0.8008412718772888,
      0.7896725535392761,
      0.774461030960083,
      0.7642484307289124,
      0.7514795660972595,
      0.7421090602874756,
      0.7312721014022827,
      0.7220489978790283,
      0.7118490934371948,
      0.7032029032707214,
      0.6948071718215942,
      0.6869843602180481,
      0.679220974445343,
      0.6712666153907776,
      0.6641452312469482,
      0.6562791466712952,
      0.6488441228866577,
      0.6418116092681885,
      0.6347957253456116,
      0.628508985042572,
      0.6214657425880432,
      0.6157000064849854,
      0.6102694869041443,
      0.6051090359687805,
      0.5997083187103271,
      0.5946723818778992,
      0.5899714231491089,
      0.5858032703399658,
      0.5816116333007812,
      0.5768241286277771,
      0.5730258822441101,
      0.569121241569519,
      0.5651571750640869,
      0.5616962313652039,
      0.5583248734474182,
      0.5549166798591614,
      0.5528585910797119,
      0.5493453145027161,
      0.5463472008705139,
      0.5436496734619141,
      0.5408833026885986,
      0.5380948185920715,
      0.5353625416755676,
      0.5328769683837891,
      0.5303449034690857,
      0.5280178189277649,
      0.5259584784507751,
      0.5234414339065552,
      0.5205557942390442,
      0.5190126895904541,
      0.5162646770477295,
      0.5137878656387329,
      0.5119324922561646,
      0.5098408460617065,
      0.5075814723968506,
      0.5061367154121399,
      0.5041857957839966,
      0.5027278661727905,
      0.5010700821876526,
      0.5005977153778076,
      0.49920907616615295,
      0.49735963344573975,
      0.4960422217845917],
     'val_accuracy': [0.1666666716337204,
      0.20000000298023224,
      0.25,
      0.30000001192092896,
      0.30000001192092896,
      0.30000001192092896,
      0.3583333194255829,
      0.375,
      0.40833333134651184,
      0.4166666567325592,
      0.5166666507720947,
      0.5666666626930237,
      0.550000011920929,
      0.6000000238418579,
      0.5833333134651184,
      0.625,
      0.6000000238418579,
      0.625,
      0.5916666388511658,
      0.6000000238418579,
      0.6000000238418579,
      0.6166666746139526,
      0.6666666865348816,
      0.6833333373069763,
      0.699999988079071,
      0.7083333134651184,
      0.7166666388511658,
      0.7083333134651184,
      0.7166666388511658,
      0.7166666388511658,
      0.7166666388511658,
      0.7416666746139526,
      0.7333333492279053,
      0.7416666746139526,
      0.7666666507720947,
      0.75,
      0.75,
      0.7583333253860474,
      0.7749999761581421,
      0.7666666507720947,
      0.7833333611488342,
      0.7833333611488342,
      0.7833333611488342,
      0.7749999761581421,
      0.7749999761581421,
      0.7833333611488342,
      0.7833333611488342,
      0.7916666865348816,
      0.7916666865348816,
      0.800000011920929,
      0.800000011920929,
      0.800000011920929,
      0.8083333373069763,
      0.8083333373069763,
      0.8083333373069763,
      0.8083333373069763,
      0.8083333373069763,
      0.8083333373069763,
      0.8083333373069763,
      0.8083333373069763,
      0.8083333373069763,
      0.8083333373069763,
      0.8083333373069763,
      0.7916666865348816,
      0.7916666865348816,
      0.7916666865348816,
      0.7916666865348816,
      0.7916666865348816,
      0.7916666865348816,
      0.7916666865348816,
      0.800000011920929,
      0.8083333373069763,
      0.8083333373069763,
      0.8083333373069763,
      0.8083333373069763,
      0.8166666626930237,
      0.8166666626930237,
      0.8166666626930237,
      0.8166666626930237,
      0.8166666626930237,
      0.8166666626930237,
      0.8166666626930237,
      0.8166666626930237,
      0.8166666626930237,
      0.8166666626930237,
      0.8166666626930237,
      0.8166666626930237,
      0.8166666626930237,
      0.8166666626930237,
      0.8166666626930237,
      0.8166666626930237,
      0.8166666626930237,
      0.8166666626930237,
      0.8166666626930237,
      0.8166666626930237,
      0.8166666626930237,
      0.8166666626930237,
      0.8166666626930237,
      0.8166666626930237,
      0.8166666626930237]}



Now visualize the loss over time using `history.history`: 


```python
# The history.history["loss"] entry is a dictionary with as many values as epochs that the
# model was trained on. 
df_loss_acc = pd.DataFrame(history.history)
df_loss= df_loss_acc[['loss','val_loss']]
df_loss.rename(columns={'loss':'train','val_loss':'validation'},inplace=True)
df_acc= df_loss_acc[['accuracy','val_accuracy']]
df_acc.rename(columns={'accuracy':'train','val_accuracy':'validation'},inplace=True)
df_loss.plot(title='Model loss',figsize=(12,8)).set(xlabel='Epoch',ylabel='Loss')
df_acc.plot(title='Model Accuracy',figsize=(12,8)).set(xlabel='Epoch',ylabel='Accuracy')
```




    [Text(0, 0.5, 'Accuracy'), Text(0.5, 0, 'Epoch')]




![png](output_43_1.png)



![png](output_43_2.png)


**Congratulations**! You've finished the assignment and built two models: One that recognizes  smiles, and another that recognizes SIGN language with almost 80% accuracy on the test set. In addition to that, you now also understand the applications of two Keras APIs: Sequential and Functional. Nicely done! 

By now, you know a bit about how the Functional API works and may have glimpsed the possibilities. In your next assignment, you'll really get a feel for its power when you get the opportunity to build a very deep ConvNet, using ResNets! 

<a name='6'></a>
## 6 - Bibliography

You're always encouraged to read the official documentation. To that end, you can find the docs for the Sequential and Functional APIs here: 

https://www.tensorflow.org/guide/keras/sequential_model

https://www.tensorflow.org/guide/keras/functional
