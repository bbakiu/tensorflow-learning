from datetime import datetime
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import Input, Model
import tensorflow as tf

data = pd.read_csv('../dataset/heart.csv')

print(data.head())
print(data.info())
print(data.describe())
print(data.describe().T)
print(data.shape)

print(data.isna().sum())
print(data['sex'].value_counts())
print(data['cp'].value_counts())

sns.countplot(x='sex', hue='target', data=data)
plt.title('Heart Disease Freq for Gender')
plt.legend(['No Disease', 'Yes Disease'])
plt.xlabel('Gender (0 - Female, 1 - Male)')
plt.ylabel('Frequency')

plt.show()

sns.countplot(x='age', hue='target', data=data)
plt.title('Heart Disease Freq for Age')
plt.legend(['No Disease', 'Yes Disease'])
plt.xlabel('Age')
plt.ylabel('Frequency')

plt.show()

plt.scatter(data['age'], data['chol'], s=200)
plt.xlabel('Age')
plt.ylabel('Cholestrol')

plt.show()

features = data.drop('target', axis=1)
target = data[['target']]

print(features.columns)

caterogical_features = features[['sex', 'fbs', 'exang', 'cp', 'slope', 'ca', 'thal', 'restecg']].copy()

numeric_features = features[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']].copy()

scaler = StandardScaler()
numeric_features = pd.DataFrame(scaler.fit_transform(numeric_features), columns=numeric_features.columns,
                                index=numeric_features.index)

processed_features = pd.concat([numeric_features, caterogical_features], axis=1, sort=False)
print(processed_features.head())
X_train, X_test, y_train, y_test = train_test_split(processed_features, target, test_size=0.15, random_state=101)


def build_model():
    inputs = Input(shape=(X_train.shape[1],))

    dense_layer1 = Dense(12, activation='relu')
    x = dense_layer1(inputs)

    dropout_layer = Dropout(0.3)
    x = dropout_layer(x)

    predictions_layer = Dense(1, activation='sigmoid')
    predictions = predictions_layer(x)

    model = Model(inputs=inputs, outputs=predictions)

    model.summary()
    model.compile(optimizer=Adam(0.001), loss=BinaryCrossentropy(),
                  metrics=['accuracy', Precision(0.5), Recall(0.5), ])

    return model


model = build_model()

dataset_train = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values))
dataset_train = dataset_train.batch(16)


num_epochs = 100

dataset_test = tf.data.Dataset.from_tensor_slices((X_test.values, y_test.values))
dataset_test = dataset_test.batch(16)

training_history = model.fit(dataset_train, epochs=num_epochs)

print(training_history.history.keys())

score = model.evaluate(X_test, y_test)

score_df = pd.Series(score, index=model.metrics_names)
print(score_df)

y_pred = model.predict(X_test)

print(y_pred[:10])

pred_results = pd.DataFrame(
    {'y_test': y_test.values.flatten(),
     'y_pred': y_pred.flatten().astype('int32')},
    index=range(len(y_pred)))

print(pred_results)
print(pd.crosstab(pred_results.y_pred, pred_results.y_test))
