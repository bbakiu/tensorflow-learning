from random import randint

import cv2
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.losses import CategoricalCrossentropy

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

print(train_images.shape, test_images.shape)

print(train_images[0])
print(train_labels[:20])


lookup = [
    'Airplane',
    'Automobile',
    'Bird',
    'Cat',
    'Deer',
    'Dog',
    'Frog',
    'Horse',
    'Ship',
    'Truck'
]


def show_img(images, labels, n_images):

    random_int = randint(0, labels.shape[0] - n_images)

    imgs, labels = images[random_int : random_int + n_images], \
                   labels[random_int : random_int +  n_images]

    _, figs = plt.subplots(1, n_images, figsize=(n_images * 3, 3))

    for fig, img, label in zip(figs, imgs, labels):
        fig.imshow(img)
        ax = fig.axes

        ax.set_title(lookup[int(label)])

        ax.title.set_fontsize(20)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


show_img(train_images, train_labels, 5)

train_dir = '../dataset/train/'
test_dir = '../dataset/test/'

i = 0

# for img, label in zip(train_images, train_labels):
#     path = train_dir + str(lookup[int(label)])
#     cv2.imwrite(os.path.join(path, str(i) + '.jpeg'), img)
#     i+=1
#     cv2.waitKey(0)
#
# i=0
# for img, label in zip(test_images, test_labels):
#     path = test_dir + str(lookup[int(label)])
#     cv2.imwrite(os.path.join(path, str(i) + '.jpeg'), img)
#     i += 1
#     cv2.waitKey(0)


train_image_generator = ImageDataGenerator(rescale=1./255)
test_image_generator = ImageDataGenerator(rescale=1./255)

batch_size = 64

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(32, 32))

test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                         directory=test_dir,
                                                         shuffle=True,
                                                         target_size=(32, 32))

sample_batch = next(train_data_gen)
print(sample_batch[0].shape)


conv_model = Sequential()
conv_model.add(Conv2D(16, (3,3), padding='same', activation='relu',
                      input_shape=sample_batch[0].shape[1:]))
conv_model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
conv_model.add(MaxPooling2D(pool_size=(2,2)))
conv_model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
conv_model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
conv_model.add(MaxPooling2D(pool_size=(2,2)))
conv_model.add(Flatten())

conv_model.add(Dense(512, activation='relu'))
conv_model.add(Dense(256, activation='relu'))
conv_model.add(Dense(10, activation='softmax'))

conv_model.compile(optimizer='adam', loss=CategoricalCrossentropy(), metrics=['accuracy'])
history = conv_model.fit(train_data_gen, epochs=50, steps_per_epoch=len(train_images) // batch_size,
                         validation_data=test_data_gen,
                         validation_steps=len(test_images)//batch_size)

print(test_images[0].shape)


def perform_test(model, img, label):
    plt.imshow(img)
    test_img = np.expand_dims(img, axis=0)
    result = model.predict(test_img)

    print('Actual Label: ', lookup[int(label)])
    print('Predicted Label: ', lookup[np.argmax(result)])


perform_test(conv_model, test_images[0], test_labels[0])
perform_test(conv_model, test_images[1], test_labels[1])
