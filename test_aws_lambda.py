import os
from dotenv import load_dotenv
import requests

load_dotenv()

url = os.getenv("lambda_url")

data = {'url': 'https://bit.ly/fried-food'}

result = requests.post(url, json=data).json()
print(result)