import numpy as np
import os
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import CSVLogger

def load_local_cifar10(path):
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    x_train = []
    y_train = []

    for i in range(1, 6):
        batch = unpickle(os.path.join(path, f'data_batch_{i}'))
        x_train.append(batch[b'data'])
        y_train.append(batch[b'labels'])

    x_train = np.concatenate(x_train).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_train = np.concatenate(y_train)

    test_batch = unpickle(os.path.join(path, 'test_batch'))
    x_test = test_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_test = np.array(test_batch[b'labels'])

    return (x_train, y_train), (x_test, y_test)

# Load the CIFAR-10 dataset
cifar10_path = 'C:\\Users\\david\\Downloads\\cifar-10-python\\cifar-10-batches-py'
(x_train, y_train), (x_test, y_test) = load_local_cifar10(cifar10_path)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert labels to one-hot encoding
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Log performance metrics
csv_logger = CSVLogger('training_log.csv', append=True, separator=';')

# Train the model with data augmentation
model.fit(datagen.flow(x_train, y_train, batch_size=64),
          epochs=50,
          validation_data=(x_test, y_test),
          callbacks=[csv_logger])

# Save the model
model.save('model_augmented.h5')
