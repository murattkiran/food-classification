from flask import Flask, request, jsonify
import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor

app = Flask('food-classifier')

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

@app.route("/predict/", methods=["POST"])
def predict():
    data = request.get_json()
    url = data['url']
    
    X = preprocessor.from_url(url)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    
    float_predictions = preds[0].tolist()
    prediction_result = dict(zip(classes, float_predictions))

    predicted_class = max(prediction_result, key=prediction_result.get)
    return jsonify({'predicted_class': predicted_class})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
