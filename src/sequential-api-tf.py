import os, datetime
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

data = pd.read_csv("../dataset/Life Expectancy Data.csv")

print(data.info())
print(data.describe())
print(data.head())
print(data.shape)
print(data.isna().sum())

countries = data['Country'].unique()
print(countries)
print(data.columns)

na_cols = ['Life expectancy ', 'Adult Mortality', 'Alcohol', 'Hepatitis B', ' BMI ',
           'Polio', 'Total expenditure','Diphtheria ',  'GDP',
           ' thinness  1-19 years', ' thinness 5-9 years',  'Population',
           'Income composition of resources']

for col in na_cols:
    for country in countries:
        data.loc[data['Country']==country, col] = data.loc[data['Country']==country, col].fillna(data[data['Country']==country][col].mean())

print(data.isna().sum())

data = data.dropna()
print(data.shape)

print(data['Status'].value_counts())
print(data['Country'].value_counts())

data.boxplot('Life expectancy ')
plt.show()

sns.boxplot(x='Status', y='Life expectancy ', data=data)
plt.xlabel('Status')
plt.ylabel('Life expectancy')
plt.show()

sns.boxplot(x='Status', y='Total expenditure', data=data)
plt.xlabel('Status')
plt.ylabel('Total expenditure')
plt.show()

data_corr = data[['Life expectancy ','Adult Mortality', 'Schooling', 'Total expenditure','Diphtheria ','GDP', 'Population']].corr()
print(data_corr)
sns.heatmap(data=data_corr, cmap='viridis', annot=True)
plt.show()

features = data.drop('Life expectancy ', axis=1)
target = data[['Life expectancy ']]

features.drop('Country', inplace=True, axis=1)
categorical_features = features['Status'].copy()
print(categorical_features.head())
categorical_features = pd.get_dummies(categorical_features, drop_first=True)
numeric_features = features.drop('Status', axis=1)
print(numeric_features.head())

scaler = StandardScaler()
numeric_features = pd.DataFrame(scaler.fit_transform(numeric_features), columns=numeric_features.columns,
                                index=numeric_features.index)

print(numeric_features.describe().T)
print(numeric_features.describe())

processed_features = pd.concat([numeric_features, categorical_features], axis=1, sort=False)
print(processed_features.head())

X_train, X_test, y_train, y_test = train_test_split(processed_features, target, test_size=0.2, random_state=101)


def build_single_layer_model():
    model = Sequential()
    model.add(Dense(32, input_shape=(X_train.shape[1],), activation='sigmoid'))
    model.add(Dense(1))
    optimier = Adam(learning_rate=0.01)

    model.compile(loss='mse', metrics=['mae','mse'], optimizer=optimier)
    return model


model = build_single_layer_model()

print(model.summary())
# print(tf.keras.utils.plot_model(model))

num_epocs = 100

training_history = model.fit(X_train, y_train, epochs=num_epocs, validation_split=0.2, verbose=True)

plt.plot(training_history.history['mae'])
plt.plot(training_history.history['val_mae'])
plt.show()

plt.plot(training_history.history['loss'])
plt.plot(training_history.history['val_loss'])
plt.show()

model.evaluate(X_test, y_test)

y_pred = model.predict(X_test)

r2_score(y_test, y_pred)
