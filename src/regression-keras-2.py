import json
import pprint

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

processed_data = pd.read_csv('../dataset/insurance_processed.csv')
print(processed_data.head())
processed_features = processed_data.drop('charges', axis=1)
target = processed_data[['charges']]

X_train, X_test, y_train, y_test = train_test_split(processed_features, target, test_size=0.2, random_state=101)

model = load_model('./models/relu_es.h5')
print(model.summary())

predictions = model.predict(X_test)
print(r2_score(y_test, predictions))