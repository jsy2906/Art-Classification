import tensorflow as tf
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

weight = 227
height = 227
channel = 3
autotune = tf.data.experimental.AUTOTUNE
seed = 42

# 데이터 경로 지정
path = './art_classification'

train = os.path.join(path, 'train')
test = os.path.join(path, 'test/0')

# trainset, valset 설정
trainset = tf.keras.preprocessing.image_dataset_from_directory(train, 
                                                           image_size=(weight, height),
                                                           validation_split=.3,
                                                           subset='training',
                                                           seed=seed, batch_size=10)
labels = trainset.class_names
trainset = trainset.cache().prefetch(autotune)

valset = tf.keras.preprocessing.image_dataset_from_directory(train, 
                                                           image_size=(weight, height),
                                                           validation_split=.3,
                                                           subset='validation',
                                                           seed=seed, batch_size=10)
valset = valset.cache().prefetch(autotune)

# 데이터 Augmentation
augmentor = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip(input_shape = (weight, height, channel)),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.3),
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
    ])

# 데이터 정규화
norm = tf.keras.layers.experimental.preprocessing.Rescaling(1/255)

# 모델 생성
vgg16 = tf.keras.applications.VGG16(include_top=False, input_shape=(weight, height, channel))
vgg16.trainable = False

model = tf.keras.Sequential([
    augmentor,
    norm,
    vgg16,
    tf.keras.layers.GlobalAvgPool2D(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(7, activation='softmax')
])

es = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience =30)

model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics=['acc'])
history = model.fit(trainset, validation_data=valset, epochs=100, callbacks=es)

# Accuracy와 Loss 그래프로 시각화
def graph_show(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Acc vs Val_Acc')
    plt.plot(epochs, acc, label = 'acc')
    plt.plot(epochs, val_acc, label = 'val_acc')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title('Loss vs Val_Loss')
    plt.plot(epochs, loss, label = 'loss')
    plt.plot(epochs, val_loss, label = 'val_loss')
    plt.legend()

# testset 설정
test_dir = os.listdir(test)
    
# testset 시각화
def im_show(file_name):
    image_dir = os.path.join(test, file_name)
    image = tf.image.decode_jpeg(tf.io.read_file(image_dir))
    plt.imshow(image)

# 결과 예측
def pred(file_name):
    image_dir = os.path.join(test, file_name)
    image = tf.image.decode_jpeg(tf.io.read_file(image_dir))
    arr = tf.keras.preprocessing.image.img_to_array(image)
    arr_exp = tf.expand_dims(arr, axis = 0)
    
    prediction = model.predict(arr_exp)
    score = tf.nn.softmax(prediction[0])
    result = np.argmax(score)
    
    return result, labels[result]
