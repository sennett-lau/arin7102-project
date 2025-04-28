import requests

url = "http://localhost:7101/api/chat"
payload = {
    "message": "请推荐适合30岁男性感冒的药物"
}
headers = {
    "Content-Type": "application/json"
}
response = requests.post(url, json=payload, headers=headers)
print(response.json())