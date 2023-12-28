import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

data = {'url': 'https://bit.ly/fried-food'}

result = requests.post(url, json=data).json()
print(result)