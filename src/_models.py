from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout


class WineClassificationModel(Model):
    def __init__(self, input_shape):
        super(WineClassificationModel, self).__init__()

        self.d1 = Dense(128, activation='relu', input_shape=[input_shape])
        self.d2 = Dense(64, activation='relu')
        self.d3 = Dense(3, activation='softmax')

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)

        return x