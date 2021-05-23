import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

mask_path = "./mask/"
nomask_path = "./nomask/"
mask_list = os.listdir(mask_path)
nomask_list = os.listdir(nomask_path)
total_len = len(mask_list) + len(nomask_list)
total_img = np.zeros((total_len, 224, 224, 3))
label = np.zeros((total_len, 1))
count = 0

for mask in mask_list:
    path = mask_path + mask
    img = load_img(path, target_size=(224, 224))
    temp = img_to_array(img)
    temp = np.expand_dims(temp, axis=0)
    temp = preprocess_input(temp)
    total_img[count, :, :, :] = temp
    label[count] = 1
    count += 1

for no_mask in nomask_list:
    path = nomask_path + no_mask
    img = load_img(path, target_size=(224, 224))
    temp = img_to_array(img)
    temp = np.expand_dims(temp, axis=0)
    temp = preprocess_input(temp)
    total_img[count, :, :, :] = temp
    label[count] = 0
    count += 1

shuffle = np.random.choice(total_len, size=total_len, replace=False)
total_img = total_img[shuffle]
label = label[shuffle]

training_img = total_img[0: int(0.8 * total_len):, :, :, :]
training_label = label[0: int(0.8 * total_len)]

validation_img = total_img[int(0.8 * total_len):, :, :, :]
validation_label = label[int(0.8 * total_len):]

model = ResNet50(input_shape=(224, 224, 3), include_top=False)
flatten = Flatten()
batch_normal = BatchNormalization()
layer1 = Dense(128, input_dim=1,  activation='relu')
layer2 = Dense(1, activation='sigmoid')
model = Sequential([model, flatten, layer1, batch_normal, layer2])
model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(training_img, training_label, epochs=10, batch_size=16, validation_data=(validation_img, validation_label))
model.save("mask_detector.h5")
