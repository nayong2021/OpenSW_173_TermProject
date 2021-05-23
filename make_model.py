import os
import numpy as np
from keras.layers.normalization_v2 import BatchNormalization
from keras_preprocessing.image import load_img, img_to_array
import tensorflow as tf
from tensorflow.python.keras.applications.resnet import preprocess_input, ResNet50
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.models import Sequential

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

# %%
# create the base pre-trained model
IMG_SHAPE = (224, 224, 3)

base_model = ResNet50(input_shape=IMG_SHAPE, weights='imagenet', include_top=False)
base_model.trainable = False
base_model.summary()
print("Number of layers in the base model: ", len(base_model.layers))

flatten_layer = Flatten()
dense_layer1 = Dense(128, activation='relu')
bn_layer1 = BatchNormalization()
dense_layer2 = Dense(1, activation=tf.nn.sigmoid)

model = Sequential([
    base_model,
    flatten_layer,
    dense_layer1,
    bn_layer1,
    dense_layer2,
])

base_learning_rate = 0.001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

model.fit(training_img, training_label, epochs=10, batch_size=16, validation_data=(validation_img, validation_label))

# save model
model.save("model.h5")