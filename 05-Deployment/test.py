import requests

url = 'http://localhost:9696/predict'

client = {
    "lead_source": "events",
    "number_of_courses_viewed": 0,
    "annual_income": 0
}

response = requests.post(url, json=client)
predictions = response.json()

print(predictions)
if predictions['converted']:
    print('customer is likely to convert')
else:
    print('customer is not likely to convert')