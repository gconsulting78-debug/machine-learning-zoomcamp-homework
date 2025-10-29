import pandas as pd
import numpy as np
import sklearn

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

import pickle

print(f'pandas=={pd.__version__}')
print(f'numpy=={np.__version__}')
print(f'sklearn=={sklearn.__version__}')

def load_data():
    data_url = 'https://raw.githubusercontent.com/alexeygrigorev/datasets/master/course_lead_scoring.csv'
    df = pd.read_csv(data_url)

    df.columns = df.columns.str.lower().str.replace(' ', '_')

    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

    for c in categorical_columns:
        df[c] = df[c].str.lower().str.replace(' ', '_')

return dfr