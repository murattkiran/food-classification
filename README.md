# Food Classification

![Food Groups](images/food-groups.png)

## Problem Description

Food classification plays a crucial role in various applications, from dietary monitoring to restaurant menu analysis. The challenge lies in accurately categorizing diverse food items into specific groups. This project aims to address the following challenges:

1. **Diverse Food Categories:** The dataset encompasses a wide range of food categories, each with its own unique visual characteristics, making accurate classification challenging.

2. **Limited Data Availability:** In some categories, obtaining a sufficient amount of labeled data for training can be a hurdle, leading to potential biases and performance issues.

3. **Model Generalization:** Achieving a model that generalizes well across different food types and variations is a key objective, considering the practical application of food classification in various scenarios.

### Dataset Overview

This dataset contains 16,643 food images grouped into 11 major food categories: **Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles-Pasta, Rice, Seafood, Soup, Vegetable-Fruit**.

The dataset is divided into three splits: **evaluation, training, and validation**, and each split includes images of the 11 food categories.

### Dataset Resource

For more details and to access the dataset, you can visit [Dataset Resource](https://www.kaggle.com/datasets/trolukovich/food11-image-dataset/data).

**Notes:**
This project utilized [Saturn Cloud](https://saturncloud.io/) for efficient cloud-based computing.
Additionally, the notebook [food-classification-model-training.ipynb](https://github.com/murattkiran/food-classification/blob/main/food-classification-model-training.ipynb) contains all the training processes for the food classification model.

## 1. EDA (Exploratory Data Analysis)

The dataset consists of sample images as shown below:
![Food Image 1](images/foodimage.png) ![Food Image 2](images/foodimage2.png)

### Loading an Image

You can use the following Python code to load an image from the dataset and convert it into a numpy array of a 3D shape. Each row of the array represents the values of the red, green, and blue color channels of one pixel in the image:

```python
import numpy as np
from tensorflow.keras.preprocessing.image import load_img

# Set the path, name, and full name of the image
path = "./training/Dessert"
name = "100.jpg"
fullname = f"{path}/{name}"

# Load the image and convert it into a numpy array
img = load_img(fullname, target_size=(299, 299))
x = np.array(img)

# Print the shape of the numpy array
print("Image Shape:", x.shape)
```

## 2. Pre-trained Convolutional Neural Networks

In this section, pre-trained convolutional neural networks have been utilized for our food classification. The Keras applications offer different pre-trained models with various architectures. The model [Xception](https://keras.io/api/applications/xception/) has been employed for this project. This model takes an input image size of `(229, 229)` and scales each image pixel between `-1` and `1`.

```python
# Create an instance of the pre-trained Xception model
model = Xception(weights='imagenet', input_shape=(229, 229, 3))
```
```python
X = np.array([x])
X.shape # Output: (1, 299, 299, 3)
X = preprocess_input(X)
pred = model.predict(X)
pred.shape # Output: (1, 1000)
decode_predictions(pred)

```
- Along with image size, the model also expects the `batch_size` which is the size of the batches of data (default 32). If one image is passed to the model, then the expected shape of the model should be (1, 229, 229, 3)
- The preprocess_input function was used on our data to make predictions, as shown in the statement: `X = preprocess_input(X)`
- The `pred = model.predict(X)` function returns 2D array of shape `(1, 1000)`, where 1000 is the probablity of the image classes. `decode_predictions(pred)` can be used to get the class names and their probabilities in readable format.

## 3. Transfer Learning

Following are the steps to create train/validation data for model:

```python
# Build image generator for training (takes preprocessing input function)
train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Load in train dataset into train generator
train_ds = train_gen.flow_from_directory('./training', # Train images directory
                                         target_size=(150,150), # resize images to train faster
                                         batch_size=32) # 32 images per batch

# Create image generator for validation
val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Load in image for validation
val_ds = val_gen.flow_from_directory('./validation', # Validation image directory
                                     target_size=(150,150),
                                     batch_size=32,
                                     shuffle=False) # False for validation
```

Following are the steps to build model from a pretrained model:

```python
# Build base model
base_model = Xception(weights='imagenet',
                      include_top=False, # to create custom dense layer
                      input_shape=(150,150,3))

# Freeze the convolutional base by preventing the weights being updated during training
base_model.trainable = False

# Define expected image shape as input
inputs = keras.Input(shape=(150,150,3))

# Feed inputs to the base model
base = base_model(inputs, training=False) # set False because the model contains BatchNormalization layer

# Convert matrices into vectors using pooling layer
vectors = keras.layers.GlobalAveragePooling2D()(base)

# Create dense layer of 10 classes
outputs = keras.layers.Dense(10)(vectors)

# Create model for training
model = keras.Model(inputs, outputs)
```
Following are the steps to instantiate optimizer and loss function:

```python
# Define learning rate
learning_rate = 0.01

# Create optimizer
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

# Define loss function
loss = keras.losses.CategoricalCrossentropy(from_logits=True) # to keep the raw output of dense layer without applying softmax

# Compile the model
model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['accuracy']) # evaluation metric accuracy
```

The model is ready to train once it is defined and compiled:

```python
# Train the model, validate it with validation data, and save the training history
history = model.fit(train_ds, epochs=15, validation_data=val_ds)
```
