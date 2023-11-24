from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPool2D, Flatten, Dense
import torch
import torch.nn as nn
import torch.nn.functional as F


class IOTSQAClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Sequential([
            Convolution2D(filters=8, kernel_size=3, padding='same', input_shape=[224, 224, 1], activation='relu',
                          name='conv_1'),
            MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool_1'),
            Convolution2D(filters=4, kernel_size=3, padding='same', activation='relu', name='conv_2'),
            MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool_2'),
            Convolution2D(filters=2, kernel_size=3, padding='same', activation='relu', name='conv_3'),
            MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool_3'),
            Convolution2D(filters=2, kernel_size=3, padding='same', activation='relu', name='conv_4'),
            MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool_4'),
            Convolution2D(filters=2, kernel_size=3, padding='same', activation='relu', name='conv_5'),
            MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool_5'),
            Flatten(name='flatten'),
            Dense(2, activation='softmax', name='softmax')
        ])
        self.model.load_weights('models/pretrained/iot_sqa_classifier.h5')

        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        orig_device = x.device
        x = x.cpu().detach().numpy().transpose(0, 2, 3, 1)
        x = self.model(x).numpy()
        x = torch.from_numpy(x).float().to(orig_device)
        return x


class IOTSQAEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Sequential([
            Convolution2D(filters=8, kernel_size=3, padding='same', input_shape=[224, 224, 1], activation='relu',
                          name='conv_1'),
            MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool_1'),
            Convolution2D(filters=4, kernel_size=3, padding='same', activation='relu', name='conv_2'),
            MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool_2'),
            Convolution2D(filters=2, kernel_size=3, padding='same', activation='relu', name='conv_3'),
            MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool_3'),
            Convolution2D(filters=2, kernel_size=3, padding='same', activation='relu', name='conv_4'),
            MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool_4'),
            Convolution2D(filters=2, kernel_size=3, padding='same', activation='relu', name='conv_5'),
            MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool_5'),
            Flatten(name='flatten'),
            Dense(128, activation='linear', name='embedding')
        ])
        self.model.load_weights('models/pretrained/iot_sqa_encoder.h5')

    def forward(self, x):
        orig_device = x.device
        x = x.cpu().detach().numpy().transpose(0, 2, 3, 1)
        x = self.model(x).numpy()
        x = torch.from_numpy(x).float().to(orig_device)
        return x
