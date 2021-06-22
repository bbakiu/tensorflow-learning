from random import randint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


data = pd.read_csv('../dataset/A_Z Handwritten Data.csv')
print(data.sample(10))
print(data.shape)

data = data.sample(frac=0.05).reset_index(drop=True)
print(data.shape)
print(data.columns)

print(sorted(data['0'].unique()))
lookup = { 0: 'A', 1: 'B', 2: 'C', 3: 'D',
           4: 'E', 5: 'F', 6: 'G', 7: 'H',
           8: 'I', 9: 'J',10: 'K', 11: 'L',
           12: 'M', 13: 'N', 14: 'O', 15: 'P',
           16: 'Q', 17: 'R', 18: 'S',19: 'T',
           20: 'U', 21: 'V', 22: 'W', 23: 'X',
           24: 'Y', 25: 'Z'}

features = data[data.columns[1:]]
target = data['0']
features = features.values.reshape(len(features), 28, 28)
print(features.shape)

print(target.loc[10])
print(features[10][10:20])


def show_image(img_features, actual_label):
    print('Actual Label: ', lookup[actual_label])
    plt.imshow(img_features, cmap='Greys')
    plt.show()


show_image(features[10], target[10])

features = features.astype(np.float32) / 255

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=101)




