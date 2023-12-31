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

## Loading an Image

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
