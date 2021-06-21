import pandas as pd
import numpy as np
import pickle
data=pd.read_csv('diabetes.data')
X=data[['BMI','Glucose','BloodPressure','SkinThickness','Insulin']]
y=data[['Outcome']]
from sklearn.linear_model import LogisticRegression
sv = LogisticRegression().fit(X,y)
pickle.dump(sv, open('dia.pkl', 'wb'))
