from tensorflow.keras.models import model_from_json
import numpy as np
from tensorflow.keras.models import load_model

model = model_from_json(open("model/facial_expression_model_structure.json", "r").read())
model.load_weights('model/facial_expression_model_weights.h5')  # load weights

#model = load_model('model/my_model.h5')


# -----------------------------

with open(r"model/fer2013.csv") as f:
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




        print(usage, emotion)
        if 'PublicTest' in usage:
            y_test.append(emotion)
            x_test.append(pixels)


    except Exception as e:

        pass



x_test = np.array(x_test, 'float32')
y_test = np.array(y_test, 'float32')

x_test /= 255

x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
x_test = x_test.astype('float32')

print(x_test.shape[0], 'test samples')
#------------------------------

Y_pred=model.predict(x_test)
print(Y_pred)
yp=[]
for i in Y_pred:
    max_index = np.argmax(i)
    yp.append(max_index)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,yp)
print("confusion matrix")
print(cm)

import numpy as np

conf_matrix = cm

accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)

print('Accuracy: {:.2f}%'.format(accuracy * 100))

import matplotlib.pyplot as plt
# calculate the confusion matrix
cm = confusion_matrix(y_test, yp)

# calculate the accuracy
accuracy = np.sum(np.diag(cm)) / np.sum(cm)

# display the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()
tick_marks = np.arange(len(emotion))
plt.xticks(tick_marks, emotion, rotation=45)
plt.yticks(tick_marks, emotion)
plt.xlabel('Predicted emotion')
plt.ylabel('True emotion')

# add labels to the cells
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, cm[i, j],
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.show()
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
score = model.evaluate(x_train, y_train, verbose=0)
print('Train loss:', score[0])
print('Train accuracy:', score[1])
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=num_epochs,
                    validation_data=(x_val, y_val),
                    verbose=1)
# plot the training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

# plot the training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

