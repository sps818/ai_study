import requests

data = {"x": "719808558,55,F,2,Graduate,Married,Less than $40K,Blue,43,2,4,3,1438.3,0,1438.3,0.707,886,27,0.421,0"}

url = "http://127.0.0.1:8080/predict"

response = requests.post(url, json=data)
print(response.json())