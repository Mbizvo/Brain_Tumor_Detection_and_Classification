from google.colab import drive
drive.mount('/content/drive')

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import to_categorical
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, GlobalMaxPooling2D, Dense,Flatten, Activation, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
tf.keras.backend.clear_session()
from tensorflow.keras.models import Model

base_path = "/content/drive/My Drive/tumor_dataset/brain_tumor"
train_path = os.path.join(base_path, "Training")
test_path = os.path.join(base_path, "Testing")

categories = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

def load_data(base_path):
    data = []
    labels = []
    for category in categories:
        category_path = os.path.join(base_path, category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (128, 128))
                data.append(img)
                labels.append(categories.index(category))
            except Exception as e:
                print(f"Error reading file: {img_path}. Error: {e}")
    return data, labels

train_data, train_labels = load_data(train_path)
test_data, test_labels = load_data(test_path)

train_data = np.array(train_data, dtype=np.float32) / 255.0
test_data = np.array(test_data, dtype=np.float32) / 255.0
print('Training shape :', train_data.shape)

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

train_data = np.expand_dims(train_data, -1)
test_data = np.expand_dims(test_data, -1)

print('Training shape :', train_data.shape)

k = len(set(train_labels))
print("number of classes: ", k)

i = Input(shape=(128, 128, 1))
x = Conv2D(32, (3, 3),strides = 2, activation='relu', padding='same')(i)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), strides = 2,activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (3, 3), strides = 2,activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), strides = 2,activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(128, (3, 3), strides = 2,activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), strides = 2,activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = GlobalMaxPooling2D()(x)

x = Dropout(0.2)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
z = Dense(k, activation='softmax')(x)

model = Model(i, z)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

batch_size = 16
r = model.fit(
    train_data, train_labels,
    validation_data=(test_data, test_labels),
    epochs=50,
    batch_size=batch_size
)
# Save the model
model.save("Brain_tumor_classification_model.h5")
print("Model saved!")

# Model Evaluation
test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

def predict_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))  # Correcting the resizing dimensions here
    img = np.array(img, dtype=np.float32) / 255.0
    img = img.reshape(1, 128, 128, 1)
    prediction = model.predict(img)
    predicted_class = categories[np.argmax(prediction)]
    return predicted_class

# Testing the prediction function
sample_image_path = "/content/drive/My Drive/tumor_dataset/brain_tumor/Testing/meningioma_tumor/image(25).jpg"
print(predict_image(sample_image_path))

from tensorflow.keras.models import load_model
model.save('/content/drive/My Drive/Models/Brain_tumor_model.h5')