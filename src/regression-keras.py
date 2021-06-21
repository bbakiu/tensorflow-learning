import datetime
import os
import json
import pprint

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

data = pd.read_csv('../dataset/insurance.csv')
print(data.describe().T)
print(data.head())
print(data.info())
print(data.isna().sum())

print(data['sex'].value_counts())
data.boxplot('charges')
plt.show()

data['charges'].plot.kde()
plt.show()

plt.scatter(x=data['age'], y=data['charges'], s=200)
plt.show()

features = data.drop('charges', axis=1)
target = data[['charges']]

print(target)
print(target.shape)

categorical_features=features[['sex', 'smoker','region']].copy()
numerical_features=features[['age', 'bmi','children']].copy()

gender_dict = {'female': 0, 'male':1}
smoker_dict = {'no': 0, 'yes':1}
categorical_features['sex'].replace(gender_dict, inplace=True)
categorical_features['smoker'].replace(smoker_dict, inplace=True)

categorical_features = pd.get_dummies(categorical_features, columns=['region'], drop_first=True)
print(categorical_features.shape)


standard_scaler = StandardScaler()
numerical_features =pd.DataFrame(standard_scaler.fit_transform(numerical_features), columns=numerical_features.columns, index=numerical_features.index)
print(numerical_features.describe().T)

processed_features = pd.concat([numerical_features, categorical_features], axis=1, sort=False)
print(processed_features.head())
processed_data = pd.concat([processed_features, target], axis=1, sort=False)

processed_data.to_csv('../dataset/insurance_processed.csv', index=False)