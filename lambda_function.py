#!/usr/bin/env python
# coding: utf-8


import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor


preprocessor = create_preprocessor('xception', target_size=(299, 299))

interpreter = tflite.Interpreter(model_path='xception_v4_48_0.886.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

classes = [
    'Bread',
    'Dairy product',
    'Dessert',
    'Egg',
    'Fried food',
    'Meat',
    'Noodles-Pasta',
    'Rice',
    'Seafood',
    'Soup',
    'Vegetable-Fruit'
]

# url = "https://bit.ly/fried-food"


def predict(url):
    X = preprocessor.from_url(url)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    float_predictions = preds[0].tolist()
    prediction_result = dict(zip(classes, float_predictions))

    return max(prediction_result, key=prediction_result.get)


def lambda_handler(event, context):
    url = event["url"]
    result = predict(url)
    return result
