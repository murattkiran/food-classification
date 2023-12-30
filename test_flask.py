import requests

url = 'http://localhost:9696/predict/'  

data = {'url': 'https://bit.ly/fried-food'}

result = requests.post(url, json=data).json()
print(result)
