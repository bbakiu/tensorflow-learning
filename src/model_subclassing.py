import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from _models import *
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

wine_data = datasets.load_wine()
print(wine_data['DESCR'])

data = pd.DataFrame(data=wine_data['data'], columns=wine_data['feature_names'])
data['target'] = wine_data['target']

print(data.head())
print(data.shape)
print(data.isna().sum())

print(data['target'].value_counts())

sns.displot(x=data['alcohol'], kde=1)
plt.show()

sns.boxplot(x='target', y='alcohol', data=data)
plt.show()

features = data.drop('target', axis=1)
target = data[['target']]

target = to_categorical(target, 3)

print(target)

standardScaler = StandardScaler()
processed_feature = pd.DataFrame(standardScaler.fit_transform(features),
                                 columns=features.columns,
                                 index=features.index)

print(processed_feature.describe().T)

X_train, X_test, y_train, y_test = train_test_split(processed_feature, target, test_size=0.2, random_state=101)

classificationModel = WineClassificationModel(X_train.shape[1])

classificationModel.compile(optimizer=SGD(lr=0.001),
                            loss=CategoricalCrossentropy(),
                            metrics=['accuracy'])

num_epochs = 500

history = classificationModel.fit(X_train.values,
                                  y_train,
                                  validation_split=0.2,
                                  epochs=num_epochs,
                                  batch_size=48)

print(history.history.keys())

plt.plot(range(num_epochs), history.history['accuracy'], label='Training Acc')
plt.show()
plt.plot(range(num_epochs), history.history['loss'], label='Training Loss')
plt.show()

predictions = classificationModel.predict(X_test)

print(predictions)
predictions = np.where(predictions >= .5, 1, predictions)
predictions = np.where(predictions < .5, 0, predictions)

print(predictions)
print(accuracy_score(y_test, predictions))
