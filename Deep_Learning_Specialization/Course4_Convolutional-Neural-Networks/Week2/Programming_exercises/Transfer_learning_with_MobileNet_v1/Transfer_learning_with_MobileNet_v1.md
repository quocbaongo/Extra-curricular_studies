# Transfer Learning with MobileNetV2

Welcome to this week's assignment, where you'll be using transfer learning on a pre-trained CNN to build an Alpaca/Not Alpaca classifier!

<img src="images/alpaca.png" style="width:300px;height:220px;">

A pre-trained model is a network that's already been trained on a large dataset and saved, which allows you to use it to customize your own model cheaply and efficiently. The one you'll be using, MobileNetV2, was designed to provide fast and computationally efficient performance. It's been pre-trained on ImageNet, a dataset containing over 14 million images and 1000 classes.

By the end of this assignment, you will be able to:

- Create a dataset from a directory
- Preprocess and augment data using the Sequential API
- Adapt a pretrained model to new data and train a classifier using the Functional API and MobileNet
- Fine-tune a classifier's final layers to improve accuracy

## Important Note on Submission to the AutoGrader

Before submitting your assignment to the AutoGrader, please make sure you are not doing the following:

1. You have not added any _extra_ `print` statement(s) in the assignment.
2. You have not added any _extra_ code cell(s) in the assignment.
3. You have not changed any of the function parameters.
4. You are not using any global variables inside your graded exercises. Unless specifically instructed to do so, please refrain from it and use the local variables instead.
5. You are not changing the assignment code where it is not required, like creating _extra_ variables.

If you do any of the following, you will get something like, `Grader Error: Grader feedback not found` (or similarly unexpected) error upon submitting your assignment. Before asking for help/debugging the errors in your assignment, check for these first. If this is the case, and you don't remember the changes you have made, you can get a fresh copy of the assignment by following these [instructions](https://www.coursera.org/learn/convolutional-neural-networks/supplement/DS4yP/h-ow-to-refresh-your-workspace).

## Table of Content

- [1 - Packages](#1)
    - [1.1 Create the Dataset and Split it into Training and Validation Sets](#1-1)
- [2 - Preprocess and Augment Training Data](#2)
    - [Exercise 1 - data_augmenter](#ex-1)
- [3 - Using MobileNetV2 for Transfer Learning](#3)
    - [3.1 - Inside a MobileNetV2 Convolutional Building Block](#3-1)
    - [3.2 - Layer Freezing with the Functional API](#3-2)
        - [Exercise 2 - alpaca_model](#ex-2)
    - [3.3 - Fine-tuning the Model](#3-3)
        - [Exercise 3](#ex-3)

<a name='1'></a>
## 1 - Packages


```python
### v2.1
```


```python
import matplotlib.pyplot as plt
import json
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.layers as tfl

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation
```

<a name='1-1'></a>
### 1.1 Create the Dataset and Split it into Training and Validation Sets

When training and evaluating deep learning models in Keras, generating a dataset from image files stored on disk is simple and fast. Call `image_data_set_from_directory()` to read from the directory and create both training and validation datasets. 

If you're specifying a validation split, you'll also need to specify the subset for each portion. Just set the training set to `subset='training'` and the validation set to `subset='validation'`.

You'll also set your seeds to match each other, so your training and validation sets don't overlap. :) 


```python
BATCH_SIZE = 32
IMG_SIZE = (160, 160)
directory = "dataset/"
train_dataset = image_dataset_from_directory(directory,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             validation_split=0.2,
                                             subset='training',
                                             seed=42)
validation_dataset = image_dataset_from_directory(directory,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             validation_split=0.2,
                                             subset='validation',
                                             seed=42)
```

    Found 327 files belonging to 2 classes.
    Using 262 files for training.
    Found 327 files belonging to 2 classes.
    Using 65 files for validation.


Now let's take a look at some of the images from the training set: 

**Note:** The original dataset has some mislabelled images in it as well.


```python
class_names = train_dataset.class_names

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
```


![png](output_9_0.png)


<a name='2'></a>
## 2 - Preprocess and Augment Training Data

You may have encountered `dataset.prefetch` in a previous TensorFlow assignment, as an important extra step in data preprocessing. 

Using `prefetch()` prevents a memory bottleneck that can occur when reading from disk. It sets aside some data and keeps it ready for when it's needed, by creating a source dataset from your input data, applying a transformation to preprocess it, then iterating over the dataset one element at a time. Because the iteration is streaming, the data doesn't need to fit into memory.

You can set the number of elements to prefetch manually, or you can use `tf.data.experimental.AUTOTUNE` to choose the parameters automatically. Autotune prompts `tf.data` to tune that value dynamically at runtime, by tracking the time spent in each operation and feeding those times into an optimization algorithm. The optimization algorithm tries to find the best allocation of its CPU budget across all tunable operations. 

To increase diversity in the training set and help your model learn the data better, it's standard practice to augment the images by transforming them, i.e., randomly flipping and rotating them. Keras' Sequential API offers a straightforward method for these kinds of data augmentations, with built-in, customizable preprocessing layers. These layers are saved with the rest of your model and can be re-used later.  Ahh, so convenient! 

As always, you're invited to read the official docs, which you can find for data augmentation [here](https://www.tensorflow.org/tutorials/images/data_augmentation).



```python
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
```

<a name='ex-1'></a>
### Exercise 1 - data_augmenter

Implement a function for data augmentation. Use a `Sequential` keras model composed of 2 layers:
* `RandomFlip('horizontal')`
* `RandomRotation(0.2)`


```python
# UNQ_C1
# GRADED FUNCTION: data_augmenter
def data_augmenter():
    '''
    Create a Sequential model composed of 2 layers
    Returns:
        tf.keras.Sequential
    '''
    ### START CODE HERE
    data_augmentation = tf.keras.Sequential()
    data_augmentation.add(RandomFlip(mode='horizontal'))
    data_augmentation.add(RandomRotation(0.2))
    ### END CODE HERE
    
    return data_augmentation
```


```python
augmenter = data_augmenter()

assert(augmenter.layers[0].name.startswith('random_flip')), "First layer must be RandomFlip"
assert augmenter.layers[0].mode == 'horizontal', "RadomFlip parameter must be horizontal"
assert(augmenter.layers[1].name.startswith('random_rotation')), "Second layer must be RandomRotation"
assert augmenter.layers[1].factor == 0.2, "Rotation factor must be 0.2"
assert len(augmenter.layers) == 2, "The model must have only 2 layers"

print('\033[92mAll tests passed!')

```

    [92mAll tests passed!


Take a look at how an image from the training set has been augmented with simple transformations:

From one cute animal, to 9 variations of that cute animal, in three lines of code. Now your model has a lot more to learn from.


```python
data_augmentation = data_augmenter()

for image, _ in train_dataset.take(1):
    plt.figure(figsize=(10, 10))
    first_image = image[0]
    #print(first_image)
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        print(augmented_image)
        plt.imshow(augmented_image[0] / 255)
        plt.axis('off')
```

    tf.Tensor(
    [[[[ 86.03226     66.77855     60.925728  ]
       [ 65.574905    50.80101     45.435913  ]
       [ 78.71769     64.95282     59.956657  ]
       ...
       [101.65427     91.32628     92.66994   ]
       [105.485405    96.39633     93.95147   ]
       [122.80664    112.119576   114.12316   ]]
    
      [[134.30685    112.44379     97.2897    ]
       [101.92154     80.73317     68.21159   ]
       [ 94.88126     77.41879     67.35187   ]
       ...
       [103.585106    93.4123      95.13521   ]
       [109.74178    100.13048     99.07811   ]
       [131.63666    118.92154    117.40343   ]]
    
      [[195.66873    173.42892    152.26053   ]
       [166.71753    143.40337    126.14525   ]
       [181.26       160.72859    147.00381   ]
       ...
       [ 96.038506    90.32874     91.75227   ]
       [104.393616    99.691025   102.36206   ]
       [113.03351    100.912796    98.36385   ]]
    
      ...
    
      [[ 20.65746      7.748315     1.8452764 ]
       [ 19.730547     7.289709     0.820184  ]
       [ 22.693514    10.191235     2.7363625 ]
       ...
       [142.81612    128.99428     93.47065   ]
       [146.86539    131.22842     96.803986  ]
       [153.4611     135.25974     99.70511   ]]
    
      [[ 16.604626     4.661518     0.9633543 ]
       [ 14.742352     3.2943482    0.47078592]
       [ 13.265074     2.9835706    0.5185572 ]
       ...
       [134.94093    123.5858      88.0406    ]
       [140.22537    126.72461     92.07448   ]
       [147.06952    131.12709     95.97269   ]]
    
      [[ 19.947685     6.1832695    0.6106552 ]
       [ 17.021267     4.7172213    0.41087997]
       [ 14.491629     3.9364543    0.22396004]
       ...
       [131.25769    120.79857     85.68161   ]
       [135.65013    123.0373      88.84338   ]
       [142.61723    127.76288     94.00337   ]]]], shape=(1, 160, 160, 3), dtype=float32)
    tf.Tensor(
    [[[[114.62754  107.77455   70.43144 ]
       [115.730125 110.08951   72.25122 ]
       [118.00404  111.52406   73.413864]
       ...
       [132.34286   94.94346   69.34452 ]
       [131.32845   95.73251   66.127884]
       [109.041145  79.98906   56.711483]]
    
      [[112.951096 107.730644  70.4104  ]
       [115.40392  110.011505  71.974075]
       [115.55019  110.82262   71.412544]
       ...
       [134.21722   96.19247   68.52143 ]
       [131.2684    95.83649   67.332924]
       [119.68689   86.35884   59.632668]]
    
      [[109.94101  105.884186  67.57223 ]
       [111.81552  107.23591   69.05275 ]
       [113.76898  109.76873   69.6998  ]
       ...
       [129.41058   94.85439   69.40046 ]
       [128.65149   96.22241   68.32103 ]
       [125.63789   92.63634   63.592525]]
    
      ...
    
      [[ 79.55377   55.273075  44.397728]
       [ 80.18649   56.189102  44.91575 ]
       [ 79.23383   54.176964  40.837418]
       ...
       [101.07799  101.18805   71.16433 ]
       [102.25654  103.07735   72.186676]
       [104.049736 104.32486   73.49682 ]]
    
      [[ 70.206955  52.338512  45.892715]
       [ 74.59239   55.922195  47.876755]
       [ 72.849464  52.90867   42.46664 ]
       ...
       [ 96.15516   97.11685   68.084274]
       [100.41638  100.650505  70.58667 ]
       [100.64499  101.40181   70.67105 ]]
    
      [[ 65.75128   50.81269   44.58348 ]
       [ 69.765884  52.16073   44.67415 ]
       [ 69.28534   51.059418  43.07951 ]
       ...
       [ 90.757164  91.86549   65.05256 ]
       [ 95.258934  94.87586   67.13992 ]
       [ 97.85422   97.10126   68.76423 ]]]], shape=(1, 160, 160, 3), dtype=float32)
    tf.Tensor(
    [[[[109.62423   78.9469    57.11194 ]
       [122.01382   87.25314   64.29448 ]
       [115.78784   80.40638   52.97965 ]
       ...
       [101.723915  97.81433   61.7807  ]
       [ 99.442535  94.462944  59.926563]
       [ 95.24013   91.09618   58.31233 ]]
    
      [[103.78743   75.392555  53.150627]
       [124.75499   90.388504  60.429974]
       [112.0588    78.393906  54.463215]
       ...
       [ 96.61234   92.31386   58.219784]
       [ 92.886894  87.35871   55.511616]
       [ 84.40254   81.91971   50.76682 ]]
    
      [[ 99.52697   69.59188   45.15857 ]
       [102.23722   72.60168   45.305748]
       [104.368965  78.126015  56.07271 ]
       ...
       [ 91.00081   86.02226   55.869175]
       [ 84.862114  80.63495   51.323963]
       [ 75.91069   73.50032   45.167156]]
    
      ...
    
      [[109.53523  102.195114  74.97662 ]
       [109.96041  101.62943   75.59395 ]
       [108.24608   96.933914  74.23258 ]
       ...
       [ 99.190285  92.347786  70.90735 ]
       [ 98.04264   87.11708   67.67212 ]
       [ 85.335396  62.28528   41.70415 ]]
    
      [[110.20137  101.452835  75.365074]
       [107.99751  100.25473   74.458984]
       [103.37886   96.50268   71.04031 ]
       ...
       [ 96.38432   90.993744  69.541756]
       [ 90.662125  83.81827   65.17864 ]
       [ 68.95942   49.640144  33.200027]]
    
      [[105.21618   98.49998   71.70924 ]
       [103.97491   96.927216  70.58064 ]
       [101.71138   94.65661   68.71474 ]
       ...
       [ 99.86667   91.832344  69.26372 ]
       [ 95.6879    85.03702   66.55903 ]
       [ 78.880455  62.500443  46.87568 ]]]], shape=(1, 160, 160, 3), dtype=float32)
    tf.Tensor(
    [[[[115.705475 113.526474  71.522156]
       [115.44516  113.96393   71.33991 ]
       [116.90455  114.31376   73.666855]
       ...
       [181.83026  165.01172  154.98135 ]
       [148.91171  130.50574  119.11644 ]
       [153.96475  139.14087  132.15216 ]]
    
      [[115.661606 114.39923   72.00316 ]
       [118.357315 114.94637   73.96551 ]
       [116.80972  113.83972   72.954056]
       ...
       [183.91296  167.56242  158.0098  ]
       [144.63965  126.45944  112.17464 ]
       [149.81009  132.3465   123.49338 ]]
    
      [[112.28495  108.07484   74.34352 ]
       [117.47     114.04683   75.31398 ]
       [117.343    114.25625   75.137115]
       ...
       [190.01608  174.91917  163.85664 ]
       [172.45538  151.55534  138.79152 ]
       [154.50482  136.17743  124.24629 ]]
    
      ...
    
      [[189.87274  159.85568  138.70319 ]
       [176.50928  142.34183  118.45593 ]
       [141.6057   108.66083   87.25603 ]
       ...
       [ 93.16644   91.24462   66.51547 ]
       [ 97.33037   93.40382   67.863174]
       [101.375206  93.865746  67.95367 ]]
    
      [[203.31812  175.12346  156.92125 ]
       [178.32092  145.36136  120.55957 ]
       [132.0184   102.560745  81.39363 ]
       ...
       [ 93.64861   90.78738   66.41698 ]
       [ 95.32535   91.03087   66.22864 ]
       [100.78612   90.85178   66.22743 ]]
    
      [[191.47972  163.37466  144.91884 ]
       [155.71696  125.46484  103.470276]
       [118.029526  93.05133   73.19112 ]
       ...
       [ 97.214935  91.13803   65.77445 ]
       [ 99.80262   90.321304  66.04751 ]
       [101.99083   90.578384  65.67064 ]]]], shape=(1, 160, 160, 3), dtype=float32)
    tf.Tensor(
    [[[[1.02699371e+02 7.29613495e+01 4.49305649e+01]
       [1.01431351e+02 6.77414551e+01 4.11940308e+01]
       [1.20735786e+02 8.52128448e+01 5.92151642e+01]
       ...
       [5.35773993e+00 3.33753633e+00 3.72238457e-01]
       [5.03981018e+00 4.91818810e+00 6.45881176e-01]
       [2.51861405e+00 3.25327563e+00 1.40722208e-02]]
    
      [[9.57934952e+01 6.50397415e+01 4.38878593e+01]
       [1.02578148e+02 7.10923691e+01 4.85054016e+01]
       [1.17229256e+02 8.24475937e+01 5.75846214e+01]
       ...
       [2.75373602e+00 1.38335252e+00 1.39486194e-01]
       [2.73889661e+00 2.10432529e+00 6.64012730e-02]
       [2.50222135e+00 3.44118595e+00 5.85541304e-04]]
    
      [[1.08262077e+02 7.46676178e+01 5.03800583e+01]
       [1.15984245e+02 8.01816711e+01 5.32135162e+01]
       [1.23076538e+02 8.53633270e+01 5.85486488e+01]
       ...
       [6.96406841e+00 5.28160048e+00 1.25391173e+00]
       [5.74916983e+00 5.89922190e+00 1.16032612e+00]
       [3.39971137e+00 5.13165951e+00 6.01765931e-01]]
    
      ...
    
      [[9.06495590e+01 8.46826706e+01 5.90028267e+01]
       [9.02377548e+01 8.44221268e+01 6.08675232e+01]
       [9.17664108e+01 8.54452438e+01 6.22338791e+01]
       ...
       [7.02957916e+01 5.53764801e+01 5.28111038e+01]
       [8.28181534e+01 6.59315109e+01 6.39720001e+01]
       [6.04700851e+01 5.08356552e+01 5.62251663e+01]]
    
      [[9.07501984e+01 8.51943512e+01 5.81300926e+01]
       [8.98852081e+01 8.48869705e+01 6.28100090e+01]
       [9.19517517e+01 8.59925995e+01 6.30355911e+01]
       ...
       [7.10333481e+01 5.54100456e+01 5.02309265e+01]
       [5.75440216e+01 4.65295486e+01 4.97700806e+01]
       [5.93246346e+01 5.20661774e+01 5.67130623e+01]]
    
      [[9.09250717e+01 8.65287094e+01 6.01369896e+01]
       [9.23628540e+01 8.79157867e+01 6.43070908e+01]
       [9.26553650e+01 8.73138733e+01 6.30744171e+01]
       ...
       [7.90963745e+01 5.95355835e+01 5.27731247e+01]
       [7.52939453e+01 6.07629852e+01 6.41174469e+01]
       [8.24867477e+01 7.35146255e+01 6.53443909e+01]]]], shape=(1, 160, 160, 3), dtype=float32)
    tf.Tensor(
    [[[[127.62523   84.31378   50.528053]
       [126.609184  84.936424  51.735012]
       [115.5262    78.19145   46.41926 ]
       ...
       [117.01191  111.60373   74.05837 ]
       [116.75708  112.28161   75.16287 ]
       [115.44748  111.541824  74.329834]]
    
      [[114.387146  74.79273   41.794685]
       [ 98.46832   63.24887   31.72157 ]
       [103.539215  66.70249   35.59337 ]
       ...
       [117.764435 110.481064  73.379105]
       [114.90526  110.85356   73.054054]
       [114.37456  111.18494   73.05383 ]]
    
      [[107.46828   69.90216   41.47894 ]
       [ 99.17995   61.30617   33.744553]
       [ 98.89734   64.170746  35.68991 ]
       ...
       [117.033066 111.166145  74.14893 ]
       [114.246765 110.441925  73.73364 ]
       [113.650536 110.28768   73.19844 ]]
    
      ...
    
      [[ 93.35445   92.7469    64.11461 ]
       [ 92.30661   91.43538   62.57058 ]
       [ 95.05497   92.38711   62.911613]
       ...
       [160.26273  144.92245  129.48825 ]
       [160.5395   143.46413  131.15927 ]
       [177.12793  161.77432  152.2449  ]]
    
      [[ 96.27404   93.382065  65.01649 ]
       [ 89.61334   89.01443   60.753384]
       [ 94.80635   93.16409   63.60611 ]
       ...
       [167.9259   146.57124  128.16574 ]
       [177.38783  156.9105   142.63095 ]
       [172.62265  156.82408  144.77823 ]]
    
      [[ 97.74545   92.29846   64.80678 ]
       [ 96.25712   92.42617   63.568123]
       [ 98.84337   95.283295  66.249954]
       ...
       [154.23532  132.08723  114.99278 ]
       [143.7324   123.95654  106.8887  ]
       [157.09782  138.49799  122.78262 ]]]], shape=(1, 160, 160, 3), dtype=float32)
    tf.Tensor(
    [[[[1.08919621e+01 7.08687544e+00 1.66306603e+00]
       [6.62154484e+00 5.27434778e+00 7.61140406e-01]
       [3.07247233e+00 3.31548929e+00 0.00000000e+00]
       ...
       [1.16495743e+02 8.28497162e+01 5.76932449e+01]
       [1.10572220e+02 7.35807495e+01 4.69142570e+01]
       [1.21178223e+02 8.69817047e+01 6.30523415e+01]]
    
      [[9.84948063e+00 6.31411934e+00 1.68058491e+00]
       [5.83550072e+00 5.02336502e+00 5.83875000e-01]
       [3.45290446e+00 4.18404484e+00 0.00000000e+00]
       ...
       [1.31523651e+02 9.49855194e+01 6.75772629e+01]
       [1.29599060e+02 9.11638794e+01 6.33014641e+01]
       [1.13350410e+02 7.78957596e+01 5.43503456e+01]]
    
      [[9.24157715e+00 6.51012897e+00 2.13142133e+00]
       [4.25871515e+00 2.91465807e+00 1.11300528e-01]
       [2.84162474e+00 2.40591240e+00 0.00000000e+00]
       ...
       [1.22295914e+02 8.65370026e+01 5.86761398e+01]
       [1.29991989e+02 9.44005203e+01 6.36142693e+01]
       [1.27981346e+02 9.48288422e+01 6.93975372e+01]]
    
      ...
    
      [[7.28646164e+01 6.26016197e+01 6.04084473e+01]
       [6.02292862e+01 4.88570557e+01 5.29924889e+01]
       [6.04980774e+01 4.50908699e+01 4.74591293e+01]
       ...
       [9.47010345e+01 9.11926422e+01 6.19892044e+01]
       [9.47504196e+01 9.07323761e+01 6.29281235e+01]
       [9.35213776e+01 8.95774689e+01 6.15108261e+01]]
    
      [[9.35429306e+01 7.81267395e+01 6.48119125e+01]
       [6.05021515e+01 5.03283234e+01 5.58688850e+01]
       [5.42733917e+01 4.35881653e+01 5.02148056e+01]
       ...
       [1.00865334e+02 9.68910904e+01 6.73722610e+01]
       [9.99540710e+01 9.57611465e+01 6.67247391e+01]
       [9.94615784e+01 9.53891525e+01 6.58435516e+01]]
    
      [[9.83250351e+01 8.56837463e+01 6.24392738e+01]
       [7.60240479e+01 6.74618378e+01 5.85251083e+01]
       [5.20070992e+01 4.77715034e+01 5.17568283e+01]
       ...
       [1.06426422e+02 1.01447083e+02 7.13667679e+01]
       [1.07469246e+02 1.02583893e+02 7.35739517e+01]
       [1.10163925e+02 1.02966072e+02 7.47840652e+01]]]], shape=(1, 160, 160, 3), dtype=float32)
    tf.Tensor(
    [[[[119.69717    85.02195    60.072342 ]
       [117.7571     86.10081    63.075077 ]
       [104.510574   70.599205   47.1772   ]
       ...
       [  8.892622    8.30912     1.221224 ]
       [  9.071436    7.5792837   1.4392127]
       [  7.861863    6.5545816   1.5164332]]
    
      [[106.79835    74.27596    49.993877 ]
       [119.11859    87.74625    64.74351  ]
       [ 91.475876   61.88263    38.564983 ]
       ...
       [ 10.616219    9.163359    1.7875979]
       [ 10.76725     9.196392    1.4459147]
       [  9.588864    7.987424    2.0355203]]
    
      [[106.524345   70.60792    46.58197  ]
       [102.33308    72.01961    48.87004  ]
       [109.034996   81.37311    56.146633 ]
       ...
       [ 13.291729   12.50262     2.9467454]
       [ 13.516655   12.306324    2.4166327]
       [ 13.284338   11.506903    3.0861866]]
    
      ...
    
      [[100.099335   93.88114    65.11242  ]
       [ 99.75159    95.35392    65.78124  ]
       [ 99.396576   93.70117    65.562126 ]
       ...
       [ 79.75282    72.32018    52.023365 ]
       [ 78.196625   71.9095     50.21616  ]
       [ 81.46613    72.24487    51.709694 ]]
    
      [[ 99.28081    94.810265   65.368416 ]
       [100.30464    95.94752    65.82407  ]
       [100.47165    95.39209    66.08334  ]
       ...
       [ 77.53833    71.063416   50.525173 ]
       [ 77.415794   70.227325   50.383152 ]
       [ 82.929924   73.67538    54.172203 ]]
    
      [[102.121796   98.15476    68.76659  ]
       [103.631714   99.585976   69.262436 ]
       [104.16037    99.725174   70.48266  ]
       ...
       [ 80.55398    72.724174   51.890797 ]
       [ 80.25478    73.66013    52.25129  ]
       [ 81.031525   73.01299    52.21375  ]]]], shape=(1, 160, 160, 3), dtype=float32)
    tf.Tensor(
    [[[[6.88175774e+00 4.37137222e+00 2.38328725e-01]
       [7.33705854e+00 6.38273811e+00 1.16330469e+00]
       [1.49861994e+01 1.19862919e+01 4.37233686e+00]
       ...
       [1.28468781e+02 9.27911377e+01 6.31080856e+01]
       [1.05789536e+02 7.27513657e+01 4.76475449e+01]
       [1.15519981e+02 7.94867249e+01 5.45798187e+01]]
    
      [[6.19540167e+00 3.26607871e+00 1.52456835e-01]
       [5.49283600e+00 5.03667927e+00 6.47497594e-01]
       [6.55077028e+00 7.36272383e+00 8.35761368e-01]
       ...
       [1.24619965e+02 9.07493362e+01 6.01040916e+01]
       [1.12847000e+02 7.85408859e+01 5.56351471e+01]
       [9.90827408e+01 6.23229561e+01 3.65716934e+01]]
    
      [[2.64372730e+00 2.10189676e+00 3.17066759e-02]
       [3.29679370e+00 2.11823940e+00 1.63323861e-02]
       [5.59894753e+00 3.43651009e+00 7.87393078e-02]
       ...
       [1.12889099e+02 7.58947754e+01 5.04459915e+01]
       [1.26400681e+02 9.17504883e+01 6.98520050e+01]
       [1.18008148e+02 8.38420868e+01 5.86399231e+01]]
    
      ...
    
      [[9.09302063e+01 7.80934296e+01 6.61372375e+01]
       [1.23409096e+02 9.62428284e+01 8.30704422e+01]
       [1.02656189e+02 7.82826004e+01 7.14785385e+01]
       ...
       [1.04975784e+02 9.99367752e+01 7.03283463e+01]
       [1.06345108e+02 1.02004677e+02 7.20739670e+01]
       [1.07762688e+02 1.03846336e+02 7.14988480e+01]]
    
      [[8.43655853e+01 7.58307114e+01 5.84438324e+01]
       [1.12259552e+02 9.06912537e+01 7.38830109e+01]
       [1.14202232e+02 8.55412979e+01 7.08917236e+01]
       ...
       [1.06538223e+02 1.01048836e+02 6.96626663e+01]
       [1.07339249e+02 1.02110336e+02 7.16192703e+01]
       [1.04721245e+02 1.01409172e+02 7.04377213e+01]]
    
      [[8.25676346e+01 7.69443588e+01 5.79129601e+01]
       [8.96205292e+01 8.08149109e+01 6.12803497e+01]
       [1.16453110e+02 9.14984436e+01 7.42452850e+01]
       ...
       [1.01043243e+02 9.94387817e+01 6.60686340e+01]
       [1.05087334e+02 9.96205292e+01 6.94808350e+01]
       [1.07779350e+02 9.97654724e+01 7.05657349e+01]]]], shape=(1, 160, 160, 3), dtype=float32)



![png](output_16_1.png)


Next, you'll apply your first tool from the MobileNet application in TensorFlow, to normalize your input. Since you're using a pre-trained model that was trained on the normalization values [-1,1], it's best practice to reuse that standard with tf.keras.applications.mobilenet_v2.preprocess_input.

<font color = 'blue'>

**What you should remember:**

* When calling image_data_set_from_directory(), specify the train/val subsets and match the seeds to prevent overlap
* Use prefetch() to prevent memory bottlenecks when reading from disk
* Give your model more to learn from with simple data augmentations like rotation and flipping.
* When using a pretrained model, it's best to reuse the weights it was trained on.


```python
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
```

<a name='3'></a>
## 3 - Using MobileNetV2 for Transfer Learning 

MobileNetV2 was trained on ImageNet and is optimized to run on mobile and other low-power applications. It's 155 layers deep (just in case you felt the urge to plot the model yourself, prepare for a long journey!) and very efficient for object detection and image segmentation tasks, as well as classification tasks like this one. The architecture has three defining characteristics:

*   Depthwise separable convolutions
*   Thin input and output bottlenecks between layers
*   Shortcut connections between bottleneck layers

<a name='3-1'></a>
### 3.1 - Inside a MobileNetV2 Convolutional Building Block

MobileNetV2 uses depthwise separable convolutions as efficient building blocks. Traditional convolutions are often very resource-intensive, and  depthwise separable convolutions are able to reduce the number of trainable parameters and operations and also speed up convolutions in two steps: 

1. The first step calculates an intermediate result by convolving on each of the channels independently. This is the depthwise convolution.

2. In the second step, another convolution merges the outputs of the previous step into one. This gets a single result from a single feature at a time, and then is applied to all the filters in the output layer. This is the pointwise convolution, or: **Shape of the depthwise convolution X Number of filters.**

<img src="images/mobilenetv2.png" style="width:650px;height:450px;">
<caption><center> <u> <font color='purple'> <b>Figure 1</b> </u><font color='purple'>  : <b>MobileNetV2 Architecture</b> <br> This diagram was inspired by the original seen <a href="https://ai.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html#:~:text=MobileNetV2%20is%20a%20significant%20improvement,object%20detection%20and%20semantic%20segmentation.">here</a>.</center></caption>

Each block consists of an inverted residual structure with a bottleneck at each end. These bottlenecks encode the intermediate inputs and outputs in a low dimensional space, and prevent non-linearities from destroying important information. 

The shortcut connections, which are similar to the ones in traditional residual networks, serve the same purpose of speeding up training and improving predictions. These connections skip over the intermediate convolutions and connect the bottleneck layers. 

Let's try to train your base model using all the layers from the pretrained model. 

Similarly to how you reused the pretrained normalization values MobileNetV2 was trained on, you'll also load the pretrained weights from ImageNet.

By specifying `weights='imagenet'`, the weights are downloaded for ImageNet. But for this lab, you'll load them locally from your workspace.


```python
IMG_SHAPE = IMG_SIZE + (3,)
base_model_path="imagenet_base_model/with_top_mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160.h5"

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=True,
                                               weights=base_model_path)
```

Print the model summary below to see all the model's layers, the shapes of their outputs, and the total number of parameters, trainable and non-trainable. 


```python
base_model.summary()
```

    Model: "mobilenetv2_1.00_160"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_22 (InputLayer)           [(None, 160, 160, 3) 0                                            
    __________________________________________________________________________________________________
    Conv1_pad (ZeroPadding2D)       (None, 161, 161, 3)  0           input_22[0][0]                   
    __________________________________________________________________________________________________
    Conv1 (Conv2D)                  (None, 80, 80, 32)   864         Conv1_pad[0][0]                  
    __________________________________________________________________________________________________
    bn_Conv1 (BatchNormalization)   (None, 80, 80, 32)   128         Conv1[0][0]                      
    __________________________________________________________________________________________________
    Conv1_relu (ReLU)               (None, 80, 80, 32)   0           bn_Conv1[0][0]                   
    __________________________________________________________________________________________________
    expanded_conv_depthwise (Depthw (None, 80, 80, 32)   288         Conv1_relu[0][0]                 
    __________________________________________________________________________________________________
    expanded_conv_depthwise_BN (Bat (None, 80, 80, 32)   128         expanded_conv_depthwise[0][0]    
    __________________________________________________________________________________________________
    expanded_conv_depthwise_relu (R (None, 80, 80, 32)   0           expanded_conv_depthwise_BN[0][0] 
    __________________________________________________________________________________________________
    expanded_conv_project (Conv2D)  (None, 80, 80, 16)   512         expanded_conv_depthwise_relu[0][0
    __________________________________________________________________________________________________
    expanded_conv_project_BN (Batch (None, 80, 80, 16)   64          expanded_conv_project[0][0]      
    __________________________________________________________________________________________________
    block_1_expand (Conv2D)         (None, 80, 80, 96)   1536        expanded_conv_project_BN[0][0]   
    __________________________________________________________________________________________________
    block_1_expand_BN (BatchNormali (None, 80, 80, 96)   384         block_1_expand[0][0]             
    __________________________________________________________________________________________________
    block_1_expand_relu (ReLU)      (None, 80, 80, 96)   0           block_1_expand_BN[0][0]          
    __________________________________________________________________________________________________
    block_1_pad (ZeroPadding2D)     (None, 81, 81, 96)   0           block_1_expand_relu[0][0]        
    __________________________________________________________________________________________________
    block_1_depthwise (DepthwiseCon (None, 40, 40, 96)   864         block_1_pad[0][0]                
    __________________________________________________________________________________________________
    block_1_depthwise_BN (BatchNorm (None, 40, 40, 96)   384         block_1_depthwise[0][0]          
    __________________________________________________________________________________________________
    block_1_depthwise_relu (ReLU)   (None, 40, 40, 96)   0           block_1_depthwise_BN[0][0]       
    __________________________________________________________________________________________________
    block_1_project (Conv2D)        (None, 40, 40, 24)   2304        block_1_depthwise_relu[0][0]     
    __________________________________________________________________________________________________
    block_1_project_BN (BatchNormal (None, 40, 40, 24)   96          block_1_project[0][0]            
    __________________________________________________________________________________________________
    block_2_expand (Conv2D)         (None, 40, 40, 144)  3456        block_1_project_BN[0][0]         
    __________________________________________________________________________________________________
    block_2_expand_BN (BatchNormali (None, 40, 40, 144)  576         block_2_expand[0][0]             
    __________________________________________________________________________________________________
    block_2_expand_relu (ReLU)      (None, 40, 40, 144)  0           block_2_expand_BN[0][0]          
    __________________________________________________________________________________________________
    block_2_depthwise (DepthwiseCon (None, 40, 40, 144)  1296        block_2_expand_relu[0][0]        
    __________________________________________________________________________________________________
    block_2_depthwise_BN (BatchNorm (None, 40, 40, 144)  576         block_2_depthwise[0][0]          
    __________________________________________________________________________________________________
    block_2_depthwise_relu (ReLU)   (None, 40, 40, 144)  0           block_2_depthwise_BN[0][0]       
    __________________________________________________________________________________________________
    block_2_project (Conv2D)        (None, 40, 40, 24)   3456        block_2_depthwise_relu[0][0]     
    __________________________________________________________________________________________________
    block_2_project_BN (BatchNormal (None, 40, 40, 24)   96          block_2_project[0][0]            
    __________________________________________________________________________________________________
    block_2_add (Add)               (None, 40, 40, 24)   0           block_1_project_BN[0][0]         
                                                                     block_2_project_BN[0][0]         
    __________________________________________________________________________________________________
    block_3_expand (Conv2D)         (None, 40, 40, 144)  3456        block_2_add[0][0]                
    __________________________________________________________________________________________________
    block_3_expand_BN (BatchNormali (None, 40, 40, 144)  576         block_3_expand[0][0]             
    __________________________________________________________________________________________________
    block_3_expand_relu (ReLU)      (None, 40, 40, 144)  0           block_3_expand_BN[0][0]          
    __________________________________________________________________________________________________
    block_3_pad (ZeroPadding2D)     (None, 41, 41, 144)  0           block_3_expand_relu[0][0]        
    __________________________________________________________________________________________________
    block_3_depthwise (DepthwiseCon (None, 20, 20, 144)  1296        block_3_pad[0][0]                
    __________________________________________________________________________________________________
    block_3_depthwise_BN (BatchNorm (None, 20, 20, 144)  576         block_3_depthwise[0][0]          
    __________________________________________________________________________________________________
    block_3_depthwise_relu (ReLU)   (None, 20, 20, 144)  0           block_3_depthwise_BN[0][0]       
    __________________________________________________________________________________________________
    block_3_project (Conv2D)        (None, 20, 20, 32)   4608        block_3_depthwise_relu[0][0]     
    __________________________________________________________________________________________________
    block_3_project_BN (BatchNormal (None, 20, 20, 32)   128         block_3_project[0][0]            
    __________________________________________________________________________________________________
    block_4_expand (Conv2D)         (None, 20, 20, 192)  6144        block_3_project_BN[0][0]         
    __________________________________________________________________________________________________
    block_4_expand_BN (BatchNormali (None, 20, 20, 192)  768         block_4_expand[0][0]             
    __________________________________________________________________________________________________
    block_4_expand_relu (ReLU)      (None, 20, 20, 192)  0           block_4_expand_BN[0][0]          
    __________________________________________________________________________________________________
    block_4_depthwise (DepthwiseCon (None, 20, 20, 192)  1728        block_4_expand_relu[0][0]        
    __________________________________________________________________________________________________
    block_4_depthwise_BN (BatchNorm (None, 20, 20, 192)  768         block_4_depthwise[0][0]          
    __________________________________________________________________________________________________
    block_4_depthwise_relu (ReLU)   (None, 20, 20, 192)  0           block_4_depthwise_BN[0][0]       
    __________________________________________________________________________________________________
    block_4_project (Conv2D)        (None, 20, 20, 32)   6144        block_4_depthwise_relu[0][0]     
    __________________________________________________________________________________________________
    block_4_project_BN (BatchNormal (None, 20, 20, 32)   128         block_4_project[0][0]            
    __________________________________________________________________________________________________
    block_4_add (Add)               (None, 20, 20, 32)   0           block_3_project_BN[0][0]         
                                                                     block_4_project_BN[0][0]         
    __________________________________________________________________________________________________
    block_5_expand (Conv2D)         (None, 20, 20, 192)  6144        block_4_add[0][0]                
    __________________________________________________________________________________________________
    block_5_expand_BN (BatchNormali (None, 20, 20, 192)  768         block_5_expand[0][0]             
    __________________________________________________________________________________________________
    block_5_expand_relu (ReLU)      (None, 20, 20, 192)  0           block_5_expand_BN[0][0]          
    __________________________________________________________________________________________________
    block_5_depthwise (DepthwiseCon (None, 20, 20, 192)  1728        block_5_expand_relu[0][0]        
    __________________________________________________________________________________________________
    block_5_depthwise_BN (BatchNorm (None, 20, 20, 192)  768         block_5_depthwise[0][0]          
    __________________________________________________________________________________________________
    block_5_depthwise_relu (ReLU)   (None, 20, 20, 192)  0           block_5_depthwise_BN[0][0]       
    __________________________________________________________________________________________________
    block_5_project (Conv2D)        (None, 20, 20, 32)   6144        block_5_depthwise_relu[0][0]     
    __________________________________________________________________________________________________
    block_5_project_BN (BatchNormal (None, 20, 20, 32)   128         block_5_project[0][0]            
    __________________________________________________________________________________________________
    block_5_add (Add)               (None, 20, 20, 32)   0           block_4_add[0][0]                
                                                                     block_5_project_BN[0][0]         
    __________________________________________________________________________________________________
    block_6_expand (Conv2D)         (None, 20, 20, 192)  6144        block_5_add[0][0]                
    __________________________________________________________________________________________________
    block_6_expand_BN (BatchNormali (None, 20, 20, 192)  768         block_6_expand[0][0]             
    __________________________________________________________________________________________________
    block_6_expand_relu (ReLU)      (None, 20, 20, 192)  0           block_6_expand_BN[0][0]          
    __________________________________________________________________________________________________
    block_6_pad (ZeroPadding2D)     (None, 21, 21, 192)  0           block_6_expand_relu[0][0]        
    __________________________________________________________________________________________________
    block_6_depthwise (DepthwiseCon (None, 10, 10, 192)  1728        block_6_pad[0][0]                
    __________________________________________________________________________________________________
    block_6_depthwise_BN (BatchNorm (None, 10, 10, 192)  768         block_6_depthwise[0][0]          
    __________________________________________________________________________________________________
    block_6_depthwise_relu (ReLU)   (None, 10, 10, 192)  0           block_6_depthwise_BN[0][0]       
    __________________________________________________________________________________________________
    block_6_project (Conv2D)        (None, 10, 10, 64)   12288       block_6_depthwise_relu[0][0]     
    __________________________________________________________________________________________________
    block_6_project_BN (BatchNormal (None, 10, 10, 64)   256         block_6_project[0][0]            
    __________________________________________________________________________________________________
    block_7_expand (Conv2D)         (None, 10, 10, 384)  24576       block_6_project_BN[0][0]         
    __________________________________________________________________________________________________
    block_7_expand_BN (BatchNormali (None, 10, 10, 384)  1536        block_7_expand[0][0]             
    __________________________________________________________________________________________________
    block_7_expand_relu (ReLU)      (None, 10, 10, 384)  0           block_7_expand_BN[0][0]          
    __________________________________________________________________________________________________
    block_7_depthwise (DepthwiseCon (None, 10, 10, 384)  3456        block_7_expand_relu[0][0]        
    __________________________________________________________________________________________________
    block_7_depthwise_BN (BatchNorm (None, 10, 10, 384)  1536        block_7_depthwise[0][0]          
    __________________________________________________________________________________________________
    block_7_depthwise_relu (ReLU)   (None, 10, 10, 384)  0           block_7_depthwise_BN[0][0]       
    __________________________________________________________________________________________________
    block_7_project (Conv2D)        (None, 10, 10, 64)   24576       block_7_depthwise_relu[0][0]     
    __________________________________________________________________________________________________
    block_7_project_BN (BatchNormal (None, 10, 10, 64)   256         block_7_project[0][0]            
    __________________________________________________________________________________________________
    block_7_add (Add)               (None, 10, 10, 64)   0           block_6_project_BN[0][0]         
                                                                     block_7_project_BN[0][0]         
    __________________________________________________________________________________________________
    block_8_expand (Conv2D)         (None, 10, 10, 384)  24576       block_7_add[0][0]                
    __________________________________________________________________________________________________
    block_8_expand_BN (BatchNormali (None, 10, 10, 384)  1536        block_8_expand[0][0]             
    __________________________________________________________________________________________________
    block_8_expand_relu (ReLU)      (None, 10, 10, 384)  0           block_8_expand_BN[0][0]          
    __________________________________________________________________________________________________
    block_8_depthwise (DepthwiseCon (None, 10, 10, 384)  3456        block_8_expand_relu[0][0]        
    __________________________________________________________________________________________________
    block_8_depthwise_BN (BatchNorm (None, 10, 10, 384)  1536        block_8_depthwise[0][0]          
    __________________________________________________________________________________________________
    block_8_depthwise_relu (ReLU)   (None, 10, 10, 384)  0           block_8_depthwise_BN[0][0]       
    __________________________________________________________________________________________________
    block_8_project (Conv2D)        (None, 10, 10, 64)   24576       block_8_depthwise_relu[0][0]     
    __________________________________________________________________________________________________
    block_8_project_BN (BatchNormal (None, 10, 10, 64)   256         block_8_project[0][0]            
    __________________________________________________________________________________________________
    block_8_add (Add)               (None, 10, 10, 64)   0           block_7_add[0][0]                
                                                                     block_8_project_BN[0][0]         
    __________________________________________________________________________________________________
    block_9_expand (Conv2D)         (None, 10, 10, 384)  24576       block_8_add[0][0]                
    __________________________________________________________________________________________________
    block_9_expand_BN (BatchNormali (None, 10, 10, 384)  1536        block_9_expand[0][0]             
    __________________________________________________________________________________________________
    block_9_expand_relu (ReLU)      (None, 10, 10, 384)  0           block_9_expand_BN[0][0]          
    __________________________________________________________________________________________________
    block_9_depthwise (DepthwiseCon (None, 10, 10, 384)  3456        block_9_expand_relu[0][0]        
    __________________________________________________________________________________________________
    block_9_depthwise_BN (BatchNorm (None, 10, 10, 384)  1536        block_9_depthwise[0][0]          
    __________________________________________________________________________________________________
    block_9_depthwise_relu (ReLU)   (None, 10, 10, 384)  0           block_9_depthwise_BN[0][0]       
    __________________________________________________________________________________________________
    block_9_project (Conv2D)        (None, 10, 10, 64)   24576       block_9_depthwise_relu[0][0]     
    __________________________________________________________________________________________________
    block_9_project_BN (BatchNormal (None, 10, 10, 64)   256         block_9_project[0][0]            
    __________________________________________________________________________________________________
    block_9_add (Add)               (None, 10, 10, 64)   0           block_8_add[0][0]                
                                                                     block_9_project_BN[0][0]         
    __________________________________________________________________________________________________
    block_10_expand (Conv2D)        (None, 10, 10, 384)  24576       block_9_add[0][0]                
    __________________________________________________________________________________________________
    block_10_expand_BN (BatchNormal (None, 10, 10, 384)  1536        block_10_expand[0][0]            
    __________________________________________________________________________________________________
    block_10_expand_relu (ReLU)     (None, 10, 10, 384)  0           block_10_expand_BN[0][0]         
    __________________________________________________________________________________________________
    block_10_depthwise (DepthwiseCo (None, 10, 10, 384)  3456        block_10_expand_relu[0][0]       
    __________________________________________________________________________________________________
    block_10_depthwise_BN (BatchNor (None, 10, 10, 384)  1536        block_10_depthwise[0][0]         
    __________________________________________________________________________________________________
    block_10_depthwise_relu (ReLU)  (None, 10, 10, 384)  0           block_10_depthwise_BN[0][0]      
    __________________________________________________________________________________________________
    block_10_project (Conv2D)       (None, 10, 10, 96)   36864       block_10_depthwise_relu[0][0]    
    __________________________________________________________________________________________________
    block_10_project_BN (BatchNorma (None, 10, 10, 96)   384         block_10_project[0][0]           
    __________________________________________________________________________________________________
    block_11_expand (Conv2D)        (None, 10, 10, 576)  55296       block_10_project_BN[0][0]        
    __________________________________________________________________________________________________
    block_11_expand_BN (BatchNormal (None, 10, 10, 576)  2304        block_11_expand[0][0]            
    __________________________________________________________________________________________________
    block_11_expand_relu (ReLU)     (None, 10, 10, 576)  0           block_11_expand_BN[0][0]         
    __________________________________________________________________________________________________
    block_11_depthwise (DepthwiseCo (None, 10, 10, 576)  5184        block_11_expand_relu[0][0]       
    __________________________________________________________________________________________________
    block_11_depthwise_BN (BatchNor (None, 10, 10, 576)  2304        block_11_depthwise[0][0]         
    __________________________________________________________________________________________________
    block_11_depthwise_relu (ReLU)  (None, 10, 10, 576)  0           block_11_depthwise_BN[0][0]      
    __________________________________________________________________________________________________
    block_11_project (Conv2D)       (None, 10, 10, 96)   55296       block_11_depthwise_relu[0][0]    
    __________________________________________________________________________________________________
    block_11_project_BN (BatchNorma (None, 10, 10, 96)   384         block_11_project[0][0]           
    __________________________________________________________________________________________________
    block_11_add (Add)              (None, 10, 10, 96)   0           block_10_project_BN[0][0]        
                                                                     block_11_project_BN[0][0]        
    __________________________________________________________________________________________________
    block_12_expand (Conv2D)        (None, 10, 10, 576)  55296       block_11_add[0][0]               
    __________________________________________________________________________________________________
    block_12_expand_BN (BatchNormal (None, 10, 10, 576)  2304        block_12_expand[0][0]            
    __________________________________________________________________________________________________
    block_12_expand_relu (ReLU)     (None, 10, 10, 576)  0           block_12_expand_BN[0][0]         
    __________________________________________________________________________________________________
    block_12_depthwise (DepthwiseCo (None, 10, 10, 576)  5184        block_12_expand_relu[0][0]       
    __________________________________________________________________________________________________
    block_12_depthwise_BN (BatchNor (None, 10, 10, 576)  2304        block_12_depthwise[0][0]         
    __________________________________________________________________________________________________
    block_12_depthwise_relu (ReLU)  (None, 10, 10, 576)  0           block_12_depthwise_BN[0][0]      
    __________________________________________________________________________________________________
    block_12_project (Conv2D)       (None, 10, 10, 96)   55296       block_12_depthwise_relu[0][0]    
    __________________________________________________________________________________________________
    block_12_project_BN (BatchNorma (None, 10, 10, 96)   384         block_12_project[0][0]           
    __________________________________________________________________________________________________
    block_12_add (Add)              (None, 10, 10, 96)   0           block_11_add[0][0]               
                                                                     block_12_project_BN[0][0]        
    __________________________________________________________________________________________________
    block_13_expand (Conv2D)        (None, 10, 10, 576)  55296       block_12_add[0][0]               
    __________________________________________________________________________________________________
    block_13_expand_BN (BatchNormal (None, 10, 10, 576)  2304        block_13_expand[0][0]            
    __________________________________________________________________________________________________
    block_13_expand_relu (ReLU)     (None, 10, 10, 576)  0           block_13_expand_BN[0][0]         
    __________________________________________________________________________________________________
    block_13_pad (ZeroPadding2D)    (None, 11, 11, 576)  0           block_13_expand_relu[0][0]       
    __________________________________________________________________________________________________
    block_13_depthwise (DepthwiseCo (None, 5, 5, 576)    5184        block_13_pad[0][0]               
    __________________________________________________________________________________________________
    block_13_depthwise_BN (BatchNor (None, 5, 5, 576)    2304        block_13_depthwise[0][0]         
    __________________________________________________________________________________________________
    block_13_depthwise_relu (ReLU)  (None, 5, 5, 576)    0           block_13_depthwise_BN[0][0]      
    __________________________________________________________________________________________________
    block_13_project (Conv2D)       (None, 5, 5, 160)    92160       block_13_depthwise_relu[0][0]    
    __________________________________________________________________________________________________
    block_13_project_BN (BatchNorma (None, 5, 5, 160)    640         block_13_project[0][0]           
    __________________________________________________________________________________________________
    block_14_expand (Conv2D)        (None, 5, 5, 960)    153600      block_13_project_BN[0][0]        
    __________________________________________________________________________________________________
    block_14_expand_BN (BatchNormal (None, 5, 5, 960)    3840        block_14_expand[0][0]            
    __________________________________________________________________________________________________
    block_14_expand_relu (ReLU)     (None, 5, 5, 960)    0           block_14_expand_BN[0][0]         
    __________________________________________________________________________________________________
    block_14_depthwise (DepthwiseCo (None, 5, 5, 960)    8640        block_14_expand_relu[0][0]       
    __________________________________________________________________________________________________
    block_14_depthwise_BN (BatchNor (None, 5, 5, 960)    3840        block_14_depthwise[0][0]         
    __________________________________________________________________________________________________
    block_14_depthwise_relu (ReLU)  (None, 5, 5, 960)    0           block_14_depthwise_BN[0][0]      
    __________________________________________________________________________________________________
    block_14_project (Conv2D)       (None, 5, 5, 160)    153600      block_14_depthwise_relu[0][0]    
    __________________________________________________________________________________________________
    block_14_project_BN (BatchNorma (None, 5, 5, 160)    640         block_14_project[0][0]           
    __________________________________________________________________________________________________
    block_14_add (Add)              (None, 5, 5, 160)    0           block_13_project_BN[0][0]        
                                                                     block_14_project_BN[0][0]        
    __________________________________________________________________________________________________
    block_15_expand (Conv2D)        (None, 5, 5, 960)    153600      block_14_add[0][0]               
    __________________________________________________________________________________________________
    block_15_expand_BN (BatchNormal (None, 5, 5, 960)    3840        block_15_expand[0][0]            
    __________________________________________________________________________________________________
    block_15_expand_relu (ReLU)     (None, 5, 5, 960)    0           block_15_expand_BN[0][0]         
    __________________________________________________________________________________________________
    block_15_depthwise (DepthwiseCo (None, 5, 5, 960)    8640        block_15_expand_relu[0][0]       
    __________________________________________________________________________________________________
    block_15_depthwise_BN (BatchNor (None, 5, 5, 960)    3840        block_15_depthwise[0][0]         
    __________________________________________________________________________________________________
    block_15_depthwise_relu (ReLU)  (None, 5, 5, 960)    0           block_15_depthwise_BN[0][0]      
    __________________________________________________________________________________________________
    block_15_project (Conv2D)       (None, 5, 5, 160)    153600      block_15_depthwise_relu[0][0]    
    __________________________________________________________________________________________________
    block_15_project_BN (BatchNorma (None, 5, 5, 160)    640         block_15_project[0][0]           
    __________________________________________________________________________________________________
    block_15_add (Add)              (None, 5, 5, 160)    0           block_14_add[0][0]               
                                                                     block_15_project_BN[0][0]        
    __________________________________________________________________________________________________
    block_16_expand (Conv2D)        (None, 5, 5, 960)    153600      block_15_add[0][0]               
    __________________________________________________________________________________________________
    block_16_expand_BN (BatchNormal (None, 5, 5, 960)    3840        block_16_expand[0][0]            
    __________________________________________________________________________________________________
    block_16_expand_relu (ReLU)     (None, 5, 5, 960)    0           block_16_expand_BN[0][0]         
    __________________________________________________________________________________________________
    block_16_depthwise (DepthwiseCo (None, 5, 5, 960)    8640        block_16_expand_relu[0][0]       
    __________________________________________________________________________________________________
    block_16_depthwise_BN (BatchNor (None, 5, 5, 960)    3840        block_16_depthwise[0][0]         
    __________________________________________________________________________________________________
    block_16_depthwise_relu (ReLU)  (None, 5, 5, 960)    0           block_16_depthwise_BN[0][0]      
    __________________________________________________________________________________________________
    block_16_project (Conv2D)       (None, 5, 5, 320)    307200      block_16_depthwise_relu[0][0]    
    __________________________________________________________________________________________________
    block_16_project_BN (BatchNorma (None, 5, 5, 320)    1280        block_16_project[0][0]           
    __________________________________________________________________________________________________
    Conv_1 (Conv2D)                 (None, 5, 5, 1280)   409600      block_16_project_BN[0][0]        
    __________________________________________________________________________________________________
    Conv_1_bn (BatchNormalization)  (None, 5, 5, 1280)   5120        Conv_1[0][0]                     
    __________________________________________________________________________________________________
    out_relu (ReLU)                 (None, 5, 5, 1280)   0           Conv_1_bn[0][0]                  
    __________________________________________________________________________________________________
    global_average_pooling2d_10 (Gl (None, 1280)         0           out_relu[0][0]                   
    __________________________________________________________________________________________________
    predictions (Dense)             (None, 1000)         1281000     global_average_pooling2d_10[0][0]
    ==================================================================================================
    Total params: 3,538,984
    Trainable params: 3,504,872
    Non-trainable params: 34,112
    __________________________________________________________________________________________________


Note the last 2 layers here. They are the so called top layers, and they are responsible of the classification in the model


```python
nb_layers = len(base_model.layers)
print(base_model.layers[nb_layers - 2].name)
print(base_model.layers[nb_layers - 1].name)
```

    global_average_pooling2d_10
    predictions


Notice some of the layers in the summary like `Conv2D` and `DepthwiseConv2D` and how they follow the progression of expansion to depthwise convolution to projection. In combination with BatchNormalization and ReLU, these make up the bottleneck layers mentioned earlier.

<font color='blue'>

**What you should remember**:

* MobileNetV2's unique features are: 
  * Depthwise separable convolutions that provide lightweight feature filtering and creation
  * Input and output bottlenecks that preserve important information on either end of the block
* Depthwise separable convolutions deal with both spatial and depth (number of channels) dimensions

Next, choose the first batch from the tensorflow dataset to use the images, and run it through the MobileNetV2 base model to test out the predictions on some of your images. 


```python
image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)
```

    (32, 1000)



```python
#Shows the different label probabilities in one tensor 
label_batch
```




    <tf.Tensor: shape=(32,), dtype=int32, numpy=
    array([1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0,
           0, 0, 1, 0, 1, 1, 1, 1, 0, 0], dtype=int32)>



Now decode the predictions made by the model. Earlier, when you printed the shape of the batch, it would have returned (32, 1000). The number 32 refers to the batch size and 1000 refers to the 1000 classes the model was pretrained on. The predictions returned by the base model below follow this format:

First the class number, then a human-readable label, and last the probability of the image belonging to that class. You'll notice that there are two of these returned for each image in the batch - these the top two probabilities returned for that image.


```python
base_model.trainable = False
image_var = tf.Variable(preprocess_input(image_batch))
pred = base_model(image_var)

# Function to decode predictions
def decode_predictions(preds, top=2):
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(class_index[str(i)]) + (pred[i],) for i in top_indices]
        results.append(result)
    return results

with open("imagenet_base_model/imagenet_class_index.json", 'r') as f:
    class_index = json.load(f)
    
decoded_predictions = decode_predictions(pred.numpy(), top=2)
print(decoded_predictions)
```

    [[('n02489166', 'proboscis_monkey', 0.10329965), ('n02102177', 'Welsh_springer_spaniel', 0.07883611)], [('n02125311', 'cougar', 0.1654676), ('n02389026', 'sorrel', 0.10764261)], [('n02437312', 'Arabian_camel', 0.2923283), ('n02437616', 'llama', 0.27713484)], [('n03944341', 'pinwheel', 0.31154886), ('n03047690', 'clog', 0.052500293)], [('n02454379', 'armadillo', 0.73107153), ('n01990800', 'isopod', 0.038719974)], [('n02437312', 'Arabian_camel', 0.25663644), ('n02422106', 'hartebeest', 0.12122728)], [('n02437616', 'llama', 0.6612557), ('n02090721', 'Irish_wolfhound', 0.23782855)], [('n02133161', 'American_black_bear', 0.82735676), ('n02134418', 'sloth_bear', 0.02925945)], [('n01518878', 'ostrich', 0.9267562), ('n02002724', 'black_stork', 0.0017766367)], [('n01518878', 'ostrich', 0.94954586), ('n02018795', 'bustard', 0.0028661634)], [('n02437616', 'llama', 0.8699833), ('n02412080', 'ram', 0.076757126)], [('n02415577', 'bighorn', 0.2429446), ('n02412080', 'ram', 0.160565)], [('n02437616', 'llama', 0.9473245), ('n02480495', 'orangutan', 0.0076571796)], [('n09428293', 'seashore', 0.48092392), ('n09421951', 'sandbar', 0.26179993)], [('n02437312', 'Arabian_camel', 0.95963204), ('n02504458', 'African_elephant', 0.0009881927)], [('n02509815', 'lesser_panda', 0.9096807), ('n02443114', 'polecat', 0.014759211)], [('n01518878', 'ostrich', 0.74165), ('n02002724', 'black_stork', 0.07205889)], [('n02437312', 'Arabian_camel', 0.49920738), ('n02412080', 'ram', 0.11842591)], [('n01518878', 'ostrich', 0.87967354), ('n02018795', 'bustard', 0.0077298395)], [('n02437616', 'llama', 0.82569915), ('n02437312', 'Arabian_camel', 0.010480011)], [('n01518878', 'ostrich', 0.9612779), ('n02410509', 'bison', 0.0013086519)], [('n02437616', 'llama', 0.636178), ('n02412080', 'ram', 0.058401026)], [('n02437616', 'llama', 0.5928003), ('n02417914', 'ibex', 0.039721698)], [('n02437616', 'llama', 0.83541703), ('n02104029', 'kuvasz', 0.048998024)], [('n03042490', 'cliff_dwelling', 0.3091509), ('n04208210', 'shovel', 0.06726616)], [('n02093647', 'Bedlington_terrier', 0.4338772), ('n02113799', 'standard_poodle', 0.4069308)], [('n02133161', 'American_black_bear', 0.97880507), ('n02132136', 'brown_bear', 0.0055297976)], [('n01518878', 'ostrich', 0.83605814), ('n02018795', 'bustard', 0.004823002)], [('n02133161', 'American_black_bear', 0.9362426), ('n02134418', 'sloth_bear', 0.007733786)], [('n03240683', 'drilling_platform', 0.04555222), ('n04146614', 'school_bus', 0.033719867)], [('n02437616', 'llama', 0.9278842), ('n02098286', 'West_Highland_white_terrier', 0.0057286685)], [('n02437616', 'llama', 0.94477594), ('n02423022', 'gazelle', 0.0054335156)]]


Uh-oh. There's a whole lot of labels here, some of them hilariously wrong, but none of them say "alpaca."

This is because MobileNet pretrained over ImageNet doesn't have the correct labels for alpacas, so when you use the full model, all you get is a bunch of incorrectly classified images.

Fortunately, you can delete the top layer, which contains all the classification labels, and create a new classification layer.

<a name='3-2'></a>
### 3.2 - Layer Freezing with the Functional API

<img src="images/snowalpaca.png" style="width:400px;height:250px;">

In the next sections, you'll see how you can use a pretrained model to modify the classifier task so that it's able to recognize alpacas. You can achieve this in three steps: 

1. Delete the top layer (the classification layer)
    * Set `include_top` in `base_model` as False
2. Add a new classifier layer
    * Train only one layer by freezing the rest of the network
    * As mentioned before, a single neuron is enough to solve a binary classification problem.
3. Freeze the base model and train the newly-created classifier layer
    * Set `base model.trainable=False` to avoid changing the weights and train *only* the new layer
    * Set training in `base_model` to False to avoid keeping track of statistics in the batch norm layer

<a name='ex-2'></a>
### Exercise 2 - alpaca_model


```python
# UNQ_C2
# GRADED FUNCTION
def alpaca_model(image_shape=IMG_SIZE, data_augmentation=data_augmenter()):
    ''' Define a tf.keras model for binary classification out of the MobileNetV2 model
    Arguments:
        image_shape -- Image width and height
        data_augmentation -- data augmentation function
    Returns:
    Returns:
        tf.keras.model
    '''
    
    
    input_shape = image_shape + (3,)
    
    ### START CODE HERE
    
    base_model_path="imagenet_base_model/without_top_mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5"
    
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                   include_top=False, # <== Important!!!!
                                                   weights=base_model_path)
    
    # freeze the base model by making it non trainable
    base_model.trainable = False 

    # create the input layer (Same as the imageNetv2 input size)
    inputs = tf.keras.Input(shape=input_shape) 
    
    # apply data augmentation to the inputs
    x = data_augmentation(inputs)
    
    # data preprocessing using the same weights the model was trained on
    x = preprocess_input(x)
    
    # set training to False to avoid keeping track of statistics in the batch norm layer
    x = base_model(x, training=False) 
    
    # add the new Binary classification layers
    # use global avg pooling to summarize the info in each channel
    x = tfl.GlobalAveragePooling2D()(x) 
    # include dropout with probability of 0.2 to avoid overfitting
    x = tfl.Dropout(rate=0.2)(x)
        
    # use a prediction layer with one neuron (as a binary classifier only needs one)
    outputs = tfl.Dense(1)(x)
    
    ### END CODE HERE
    
    model = tf.keras.Model(inputs, outputs)
    
    return model
```

Create your new model using the data_augmentation function defined earlier.


```python
model2 = alpaca_model(IMG_SIZE, data_augmentation)
```


```python
from test_utils import summary, comparator

alpaca_summary = [['InputLayer', [(None, 160, 160, 3)], 0],
                    ['Sequential', (None, 160, 160, 3), 0],
                    ['TensorFlowOpLayer', [(None, 160, 160, 3)], 0],
                    ['TensorFlowOpLayer', [(None, 160, 160, 3)], 0],
                    ['Functional', (None, 5, 5, 1280), 2257984],
                    ['GlobalAveragePooling2D', (None, 1280), 0],
                    ['Dropout', (None, 1280), 0, 0.2],
                    ['Dense', (None, 1), 1281, 'linear']] #linear is the default activation

comparator(summary(model2), alpaca_summary)

for layer in summary(model2):
    print(layer)
    
```

    [32mAll tests passed![0m
    ['InputLayer', [(None, 160, 160, 3)], 0]
    ['Sequential', (None, 160, 160, 3), 0]
    ['TensorFlowOpLayer', [(None, 160, 160, 3)], 0]
    ['TensorFlowOpLayer', [(None, 160, 160, 3)], 0]
    ['Functional', (None, 5, 5, 1280), 2257984]
    ['GlobalAveragePooling2D', (None, 1280), 0]
    ['Dropout', (None, 1280), 0, 0.2]
    ['Dense', (None, 1), 1281, 'linear']


The base learning rate has been set for you, so you can go ahead and compile the new model and run it for 5 epochs:


```python
base_learning_rate = 0.001
model2.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
```


```python
initial_epochs = 5
history = model2.fit(train_dataset, validation_data=validation_dataset, epochs=initial_epochs)
```

    Epoch 1/5
    9/9 [==============================] - 9s 1s/step - loss: 0.7800 - accuracy: 0.5115 - val_loss: 0.7849 - val_accuracy: 0.4923
    Epoch 2/5
    9/9 [==============================] - 8s 857ms/step - loss: 0.6234 - accuracy: 0.6107 - val_loss: 0.6222 - val_accuracy: 0.5385
    Epoch 3/5
    9/9 [==============================] - 8s 867ms/step - loss: 0.5361 - accuracy: 0.7405 - val_loss: 0.5423 - val_accuracy: 0.5846
    Epoch 4/5
    9/9 [==============================] - 8s 868ms/step - loss: 0.4864 - accuracy: 0.7099 - val_loss: 0.5023 - val_accuracy: 0.6000
    Epoch 5/5
    9/9 [==============================] - 8s 869ms/step - loss: 0.4865 - accuracy: 0.7328 - val_loss: 0.3813 - val_accuracy: 0.7385


Plot the training and validation accuracy:


```python
acc = [0.] + history.history['accuracy']
val_acc = [0.] + history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
```


![png](output_45_0.png)



```python
class_names
```




    ['alpaca', 'not alpaca']



The results are ok, but could be better. Next, try some fine-tuning.

<a name='3-3'></a>
### 3.3 - Fine-tuning the Model

You could try fine-tuning the model by re-running the optimizer in the last layers to improve accuracy. When you use a smaller learning rate, you take smaller steps to adapt it a little more closely to the new data. In transfer learning, the way you achieve this is by unfreezing the layers at the end of the network, and then re-training your model on the final layers with a very low learning rate. Adapting your learning rate to go over these layers in smaller steps can yield more fine details - and higher accuracy.

The intuition for what's happening: when the network is in its earlier stages, it trains on low-level features, like edges. In the later layers, more complex, high-level features like wispy hair or pointy ears begin to emerge. For transfer learning, the low-level features can be kept the same, as they have common features for most images. When you add new data, you generally want the high-level features to adapt to it, which is rather like letting the network learn to detect features more related to your data, such as soft fur or big teeth. 

To achieve this, just unfreeze the final layers and re-run the optimizer with a smaller learning rate, while keeping all the other layers frozen.

Where the final layers actually begin is a bit arbitrary, so feel free to play around with this number a bit. The important takeaway is that the later layers are the part of your network that contain the fine details (pointy ears, hairy tails) that are more specific to your problem.

First, unfreeze the base model by setting `base_model.trainable=True`, set a layer to fine-tune from, then re-freeze all the layers before it. Run it again for another few epochs, and see if your accuracy improved!

<a name='ex-3'></a>
### Exercise 3


```python
print(model2.summary())
```

    Model: "functional_14"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_24 (InputLayer)        [(None, 160, 160, 3)]     0         
    _________________________________________________________________
    sequential_17 (Sequential)   (None, 160, 160, 3)       0         
    _________________________________________________________________
    tf_op_layer_RealDiv_7 (Tenso [(None, 160, 160, 3)]     0         
    _________________________________________________________________
    tf_op_layer_Sub_7 (TensorFlo [(None, 160, 160, 3)]     0         
    _________________________________________________________________
    mobilenetv2_1.00_160 (Functi (None, 5, 5, 1280)        2257984   
    _________________________________________________________________
    global_average_pooling2d_11  (None, 1280)              0         
    _________________________________________________________________
    dropout_7 (Dropout)          (None, 1280)              0         
    _________________________________________________________________
    dense_7 (Dense)              (None, 1)                 1281      
    =================================================================
    Total params: 2,259,265
    Trainable params: 1,281
    Non-trainable params: 2,257,984
    _________________________________________________________________
    None



```python
# UNQ_C3
base_model = model2.layers[4]
base_model.trainable = True
# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 120

### START CODE HERE

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = None
    
# Define a BinaryCrossentropy loss function. Use from_logits=True
loss_function=tf.keras.losses.BinaryCrossentropy(from_logits=True)
# Define an Adam optimizer with a learning rate of 0.1 * base_learning_rate
optimizer = tf.keras.optimizers.Adam(lr=0.1*base_learning_rate)
# Use accuracy as evaluation metric
metrics=['accuracy']

### END CODE HERE

model2.compile(loss=loss_function,
              optimizer = optimizer,
              metrics=metrics)
```

    Number of layers in the base model:  155



```python
assert type(loss_function) == tf.python.keras.losses.BinaryCrossentropy, "Not the correct layer"
assert loss_function.from_logits, "Use from_logits=True"
assert type(optimizer) == tf.keras.optimizers.Adam, "This is not an Adam optimizer"
assert optimizer.lr == base_learning_rate / 10, "Wrong learning rate"
assert metrics[0] == 'accuracy', "Wrong metric"

print('\033[92mAll tests passed!')

```

    [92mAll tests passed!



```python
fine_tune_epochs = 5
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model2.fit(train_dataset,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=validation_dataset)
```

    Epoch 5/10
    9/9 [==============================] - 10s 1s/step - loss: 0.5022 - accuracy: 0.7672 - val_loss: 0.4141 - val_accuracy: 0.7385
    Epoch 6/10
    9/9 [==============================] - 10s 1s/step - loss: 0.3154 - accuracy: 0.8435 - val_loss: 0.2421 - val_accuracy: 0.9385
    Epoch 7/10
    9/9 [==============================] - 9s 1s/step - loss: 0.2055 - accuracy: 0.9122 - val_loss: 0.1821 - val_accuracy: 0.9538
    Epoch 8/10
    9/9 [==============================] - 9s 1s/step - loss: 0.2467 - accuracy: 0.8779 - val_loss: 0.1310 - val_accuracy: 0.9692
    Epoch 9/10
    9/9 [==============================] - 9s 1s/step - loss: 0.1739 - accuracy: 0.9351 - val_loss: 0.1902 - val_accuracy: 0.8923
    Epoch 10/10
    9/9 [==============================] - 9s 1s/step - loss: 0.1417 - accuracy: 0.9542 - val_loss: 0.1425 - val_accuracy: 0.9231


Ahhh, quite an improvement! A little fine-tuning can really go a long way.


```python
acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']
```


```python
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
```


![png](output_56_0.png)


<font color='blue'>

**What you should remember**:

* To adapt the classifier to new data: Delete the top layer, add a new classification layer, and train only on that layer
* When freezing layers, avoid keeping track of statistics (like in the batch normalization layer)
* Fine-tune the final layers of your model to capture high-level details near the end of the network and potentially improve accuracy 

## Congratulations!

You've completed this assignment on transfer learning and fine-tuning. Here's a quick recap of all you just accomplished:

* Created a dataset from a directory
* Augmented data with the Sequential API
* Adapted a pretrained model to new data with the Functional API and MobileNetV2
* Fine-tuned the classifier's final layers and boosted the model's accuracy

That's awesome! 
