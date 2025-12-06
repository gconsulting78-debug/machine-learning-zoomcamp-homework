import requests

url = 'http://localhost:9696/predict'
# url = 'https://mlzoomcamp-flask-uv.fly.dev/predict'

teacher = {'teacher_ethinicity' : 'chinese',
           'teacher_age' : 24,
           'teacher_tenure' : 9,
           'student_ratio' : 27,
           'education' : 'ng',
           'teacher_rating' : 5,
           'teacher_rating_last_year' : 3, 	
           'sick_days' : 10,
           'marital_status' : 'single',
           'gender' : 'female',
           'student_grade' : 'primary',
           'subject' : 'english'
}

response = requests.post(url, json=teacher)

predictions = response.json()

if predictions['churn']:
    print('teacher is likely to churn, intervention')
else:
    print('teacher is not likely to churn')