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

categorical_features = features[['sex', 'smoker', 'region']].copy()
numerical_features = features[['age', 'bmi', 'children']].copy()

gender_dict = {'female': 0, 'male': 1}
smoker_dict = {'no': 0, 'yes': 1}
categorical_features['sex'].replace(gender_dict, inplace=True)
categorical_features['smoker'].replace(smoker_dict, inplace=True)

categorical_features = pd.get_dummies(categorical_features, columns=['region'], drop_first=True)
print(categorical_features.shape)

standard_scaler = StandardScaler()
numerical_features = pd.DataFrame(standard_scaler.fit_transform(numerical_features), columns=numerical_features.columns,
                                  index=numerical_features.index)
print(numerical_features.describe().T)

processed_features = pd.concat([numerical_features, categorical_features], axis=1, sort=False)
print(processed_features.head())
processed_data = pd.concat([processed_features, target], axis=1, sort=False)

processed_data.to_csv('../dataset/insurance_processed.csv', index=False)

X_train, X_test, y_train, y_test = train_test_split(processed_features, target, test_size=0.2, random_state=101)

model = Sequential()
model.add(Dense(32, activation='relu', input_shape=[len(X_train.keys())]))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

optimizer = Adam(0.001)
model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

print(model.summary())

epochs = 500
# history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.2, verbose=True )
#
# history_df = pd.DataFrame(history.history)
# history_df['epoch'] = history.epoch
#
# print(history_df.tail())
#
# predictions = model.predict(X_test)
# print(predictions)
# print(predictions.flatten())
#
# plt.scatter(y_test, predictions.flatten(), s=200, c='darkblue')
# plt.show()
#
# print(r2_score(y_test, predictions ))

model_elu = Sequential()
model_elu.add(Dense(32, activation='elu', input_shape=[len(X_train.keys())]))
model_elu.add(Dropout(0.2))
model_elu.add(Dense(64, activation='elu'))
model_elu.add(Dense(1))

optimizer = Adam(0.001)
model_elu.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

print(model_elu.summary())

# history = model_elu.fit(X_train, y_train, epochs=epochs, validation_split=0.2, verbose=True, callbacks=[tfdocs.modeling.EpochDots()])
#
# history_df = pd.DataFrame(history.history)
# history_df['epoch'] = history.epoch
#
# print(history_df.tail())
#
# predictions = model_elu.predict(X_test).flatten()
# plt.scatter(y_test, predictions, s=200, c='darkblue')
# plt.show()
#
# print(r2_score(y_test, predictions))


model_elu_ES = Sequential()
model_elu_ES.add(Dense(32, activation='elu', input_shape=[len(X_train.keys())]))
model_elu_ES.add(Dropout(0.2))
model_elu_ES.add(Dense(64, activation='elu'))
model_elu_ES.add(Dense(1))

optimizer = Adam(0.001)
model_elu_ES.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

print(model_elu_ES.summary())

early_stop = EarlyStopping(monitor='val_loss', patience=5)

history = model_elu_ES.fit(X_train, y_train, epochs=epochs, validation_split=0.2, verbose=True,
                           callbacks=[early_stop, tfdocs.modeling.EpochDots()])

history_df = pd.DataFrame(history.history)
history_df['epoch'] = history.epoch

print(history_df.tail())

predictions = model_elu_ES.predict(X_test).flatten()
plt.scatter(y_test, predictions, s=200, c='darkblue')
plt.show()

print(r2_score(y_test, predictions))

model_elu_ES.save('./models/relu_es.h5', save_format='h5')
model_elu_ES.save_weights('./models/relu_es_weights.h5', save_format='h5')

print(pprint.pprint(json.loads(model_elu_ES.to_json())))
