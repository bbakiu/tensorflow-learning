import os, datetime
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

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
