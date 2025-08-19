import requests

url = 'http://embeddings:8080/embed'
data = {"inputs": ["What is Deep Learning?", "What is Love?"]}

x = requests.post(url, json = myobj)

print(x.text)
