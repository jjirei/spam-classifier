import requests

url = "http://127.0.0.1:5000/predict"
headers = {"Content-Type": "application/json"}

data = {"message": "Congratulations! You’ve won a prize."}
response = requests.post(url, json=data, headers=headers)

print("Raw response:", response.text)
print("Status code:", response.status_code)
print("Headers:", response.headers)

if response.headers.get("Content-Type") == "application/json":
    print(response.json())
else:
    print("Not JSON. Maybe an error or HTML?")
