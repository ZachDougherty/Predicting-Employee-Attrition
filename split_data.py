import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('attrition.csv')

y = df.Attrition.values
X = df.drop([
		'EmployeeCount',
		'Over18',
		'EmployeeNumber',
		'StandardHours',
		'Attrition'
	], axis=1)

# ensure no missing values
assert all(X.isna().sum() == np.zeros(X.shape[1]))


X_train, X_test, y_train, y_test = \
	train_test_split(X, y, train_size=0.8)

X_train.insert(X.shape[1], 'Attrition', y_train)
X_train.to_csv('attrition_train.csv')

X_test.insert(X.shape[1], 'Attrition', y_test)
X_test.to_csv('attrition_test.csv')