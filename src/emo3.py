import os

import tensorflow as tf

import keras
from keras.engine.saving import load_model

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import numpy as np

# ------------------------------
sess = tf.Session()
keras.backend.set_session(sess)
# ------------------------------
# variables
num_classes = 7  # angry, disgust, fear, happy, sad, surprise, neutral
batch_size = 256
epochs = 25
# ------------------------------

with open(
        r"model/fer2013.csv") as f:
    content = f.readlines()

lines = np.array(content)

no_instances = lines.size

# ------------------------------
# initialize trainset and test set
x_train, y_train, x_test, y_test = [], [], [], []

# ------------------------------
# transfer train and test set data
for i in range(1, no_instances):
    try:
        emotion, img, usage = lines[i].split(",")

        val = img.split(" ")

        pixels = np.array(val, 'float32')

        emotion = keras.utils.to_categorical(emotion, num_classes)

        if 'Training' in usage:
            y_train.append(emotion)
            x_train.append(pixels)
        elif 'PublicTest' in usage:
            y_test.append(emotion)
            x_test.append(pixels)
    except:
        pass

# ------------------------------
# data transformation for train and test sets
x_train = np.array(x_train, 'float32')
y_train = np.array(y_train, 'float32')
x_test = np.array(x_test, 'float32')
y_test = np.array(y_test, 'float32')

x_train /= 255  # normalize inputs between [0, 1]
x_test /= 255

x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
x_train = x_train.astype('float32')
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
x_test = x_test.astype('float32')

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# ------------------------------
# construct CNN structure
model = Sequential()

# 1st convolution layer
model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

# 2nd convolution layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

# 3rd convolution layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))


model.add(Flatten())

# fully connected neural networks
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='softmax'))
# ------------------------------
# batch process
gen = ImageDataGenerator()
train_generator = gen.flow(x_train, y_train, batch_size=batch_size)

# ------------------------------

model.compile(loss='categorical_crossentropy'
              , optimizer=keras.optimizers.Adam()
              , metrics=['accuracy']
              )

# ------------------------------

if not os.path.exists("model.h5"):

    model.fit_generator(train_generator, steps_per_epoch=batch_size, epochs=epochs)


    model.save("model.h5")  # train for randomly selected one
else:
    model = load_model("model.h5")  # load weights


# ------------------------------
#
# predict labels for the test set
y_pred = model.predict(x_test)

# convert the predicted labels from one-hot encoding to integers
y_pred_int = np.argmax(y_pred, axis=1)

# convert the true labels from one-hot encoding to integers
y_true_int = np.argmax(y_test, axis=1)

# calculate the confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true_int, y_pred_int)

# calculate the evaluation metrics
tp = np.diag(cm)
tn = np.diag(cm)[::-1]
fp = np.sum(cm, axis=0) - tp
fn = np.sum(cm, axis=1) - tp

# print the confusion matrix and evaluation metrics
print("Confusion Matrix:\n", cm)
print("True Positive:", tp)
print("True Negative:", tn)
print("False Positive:", fp)
print("False Negative:", fn)

import matplotlib.pyplot as plt

# plot the evaluation metrics
fig, ax = plt.subplots()
ax.bar(np.arange(num_classes), tp, label='True Positive')
ax.bar(np.arange(num_classes), tn, label='True Negative')
ax.bar(np.arange(num_classes), fp, bottom=tp, label='False Positive')
ax.bar(np.arange(num_classes), fn, bottom=tp + fp, label='False Negative')
ax.set_xticks(np.arange(num_classes))
ax.set_xticklabels(['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'])
ax.legend()
ax.set_title('Evaluation Metrics')
ax.set_xlabel('Emotion')
ax.set_ylabel('Count')
plt.show()


accuracy = np.sum(np.diag(cm)) / np.sum(cm)

print('Accuracy: {:.2f}%'.format(accuracy * 100))