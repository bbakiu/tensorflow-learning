import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import RMSprop

data = pd.read_csv('../dataset/Dataset_spine.csv', skiprows=1, names=['pelvic_incidence',
                                                                      'pelvic tilt',
                                                                      'lumbar_lordosis_angle',
                                                                      'sacral_slope',
                                                                      'pelvic_radius',
                                                                      'degree_spondylolisthesis',
                                                                      'pelvic_slope',
                                                                      'direct_tilt',
                                                                      'thoracic_slope',
                                                                      'cervical_tilt',
                                                                      'sacrum_angle',
                                                                      'scoliosis_slope',
                                                                      'class_att'])
print(data.head())
print(data.shape)

print(data.columns)
print(data['class_att'].unique())

sns.countplot(x='class_att', data=data)
plt.show()

sns.boxplot(x='class_att', y='pelvic_radius', data=data)
plt.show()

class_att_dict = {'Abnormal': 0, 'Normal': 1}
data['class_att'].replace(class_att_dict, inplace=True)

features = data.drop('class_att', axis=1)
target = data[['class_att']]

standard_scaler = StandardScaler()
features = pd.DataFrame(standard_scaler.fit_transform(features), columns=features.columns,
                        index=features.index)
print(features.describe().T)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.15, random_state=101)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=101)


def build_and_compile_model():
    inputs = Input(shape=(X_train.shape[1],))
    x = Dense(16, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(8, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=predictions)
    print(model.summary())
    model.compile(optimizer=RMSprop(0.001),
                  loss=BinaryCrossentropy(),
                  metrics=['accuracy', Precision(0.5), Recall(0.5)])
    return model


model = build_and_compile_model()

dataset_train = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values))
dataset_train = dataset_train.batch(16)
dataset_train.shuffle(128)

epochs = 100

model.fit(dataset_train, epochs=epochs)

dataset_val = tf.data.Dataset.from_tensor_slices((X_val.values, y_val.values))
dataset_val = dataset_train.batch(16)

# model.fit(dataset_train, epochs=epochs, validation_data=dataset_val)
predictions = model.predict(X_test)

predictions = np.where(predictions >= 0.5, 1, predictions)
predictions = np.where(predictions < 0.5, 0, predictions)

print(predictions)
