import pickle

import pandas as pd
import numpy as np
import sklearn
import xgboost as xgb

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


print(f'pandas=={pd.__version__}')
print(f'numpy=={np.__version__}')
print(f'sklearn=={sklearn.__version__}')
print(f'xgboost=={xgboost.__version__}')

def load_data():
    data_url = 'https://raw.githubusercontent.com/gconsulting78-debug/machine-learning-zoomcamp-midterm_project/refs/heads/main/Teacher-Churn_Mid_Term_Project1.csv'

    df = pd.read_csv(data_url)

    df.columns = df.columns.str.lower().str.replace(' ', '_')

    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

    for c in categorical_columns:
        df[c] = df[c].str.lower().str.replace(' ', '_')

    

    df.churn = (df.churn == 'yes').astype(int)
    return df

def train_model(df):
    numerical = ['teacher_age', 'teacher_tenure', 'student_ratio',
       'teacher_rating','teacher_rating_last_year', 'sick_days']
    categorical = ['teacher_ethinicity','education','marital_status','gender','student_grade','subject']

    df[numerical] = df[numerical].fillna(0)
    df[categorical] = df[categorical].fillna('NA')

    y_train = df.churn
    train_dict = df[categorical + numerical].to_dict(orient='records')

    pipeline = make_pipeline(
        DictVectorizer(),
        xgb.XGBClassifier(
        objective='binary:logistic',  # For binary classification
        eval_metric='logloss',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42)
    )

    pipeline.fit(train_dict, y_train)

    return pipeline

def save_model(pipeline, output_file):
    with open(output_file, 'wb') as f_out:
        pickle.dump(pipeline, f_out)

df = load_data()
pipeline = train_model(df)
save_model(pipeline, 'model.bin')

print('Model saved to model.bin')
