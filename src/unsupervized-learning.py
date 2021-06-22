from random import randint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D, MaxPool2D, Conv2DTranspose
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop, SGD

data = pd.read_csv('../dataset/A_Z Handwritten Data.csv')
print(data.sample(10))
print(data.shape)

data = data.sample(frac=0.05).reset_index(drop=True)
print(data.shape)
print(data.columns)

print(sorted(data['0'].unique()))
lookup = {0: 'A', 1: 'B', 2: 'C', 3: 'D',
          4: 'E', 5: 'F', 6: 'G', 7: 'H',
          8: 'I', 9: 'J', 10: 'K', 11: 'L',
          12: 'M', 13: 'N', 14: 'O', 15: 'P',
          16: 'Q', 17: 'R', 18: 'S', 19: 'T',
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

train_images, test_images, train_labels, test_labels = train_test_split(features, target, test_size=0.2,
                                                                        random_state=101)

stacked_encoder = Sequential()
stacked_encoder.add(Flatten(input_shape=[28, 28]))
stacked_encoder.add(Dense(64, activation='relu'))
stacked_encoder.add(Dense(32, activation='relu'))
stacked_encoder.add(Dense(16, activation='relu'))

stacked_decoder = Sequential()
stacked_decoder.add(Dense(32, activation='relu', input_shape=[16]))
stacked_decoder.add(Dense(64, activation='relu'))
stacked_decoder.add(Dense(28 * 28, activation='relu'))
stacked_decoder.add(Reshape([28, 28]))

autoencoder_model = Sequential([stacked_encoder, stacked_decoder])

autoencoder_model.compile(loss='mean_squared_error', optimizer=RMSprop(), metrics=['mse'])

history = autoencoder_model.fit(train_images, train_images, epochs=20, verbose=True)


def reconstruct_img(model, images, n_imgs):
    random_int = randint(0, images.shape[0] - n_imgs)

    reconstructions = model.predict(images[random_int: random_int + n_imgs])

    fig = plt.figure(figsize=(n_imgs * 3, 3))

    for img_index in range(n_imgs):
        plt.subplot(2, n_imgs, 1 + img_index)
        plt.imshow(images[random_int + img_index], cmap='Greys')

        plt.subplot(2, n_imgs, 1 + n_imgs + img_index)
        plt.imshow(reconstructions[img_index], cmap='Greys')

    plt.show()


reconstruct_img(autoencoder_model, test_images, 5)

conv_encoder = Sequential()

conv_encoder.add(Reshape([28, 28, 1], input_shape=[28, 28]))

conv_encoder.add(Conv2D(16, kernel_size=3, padding="SAME", activation='relu'))
conv_encoder.add(MaxPool2D(pool_size=2))

conv_encoder.add(Conv2D(32, kernel_size=3, padding="SAME", activation='relu'))
conv_encoder.add(MaxPool2D(pool_size=2))

conv_encoder.add(Conv2D(64, kernel_size=3, padding="SAME", activation='relu'))
conv_encoder.add(MaxPool2D(pool_size=2))

conv_decoder = Sequential()

conv_decoder.add(
    Conv2DTranspose(32, kernel_size=3, padding="VALID", strides=2, activation='relu', input_shape=[3, 3, 64]))

conv_decoder.add(Conv2DTranspose(16, kernel_size=3, padding="VALID", strides=2, activation='relu'))

conv_decoder.add(Conv2DTranspose(1, kernel_size=3, padding="VALID", strides=2, activation='sigmoid'))
conv_decoder.add(Reshape([28, 28]))

conv_ae_model = Sequential([conv_encoder, conv_decoder])

conv_ae_model.compile(loss=BinaryCrossentropy, optimizer=SGD(lr=1.0), metrics=['mse'])

history_conv = conv_ae_model.fit(train_images, train_images, epochs=20, verbose=True)

reconstruct_img(conv_ae_model, test_images, 5)
reconstruct_img(conv_ae_model, test_images, 5)
