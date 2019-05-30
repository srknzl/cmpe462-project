from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.models import Sequential
from keras.utils import to_categorical, plot_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import cv2
import subprocess 
import os 
import numpy as np
import matplotlib.pyplot as plt
import keras 
train_datagen = ImageDataGenerator(
    rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'animal_database',
        target_size=(250, 250),
        batch_size=32,
        class_mode='categorical')

test_generator = test_gen.flow_from_directory(
        'test_set',
        target_size=(250, 250),
        batch_size=1,
        class_mode='categorical')

model = Sequential()
model.add(Conv2D(32, (3,3) , input_shape=(250,250,3)))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3,3)))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (3,3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.1))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(19, activation='softmax'))
sgd = SGD(lr=0.001)

model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy']
)
plot_model(model, to_file='model.png',show_shapes=True)

def learningRateScheduler(index):
    if (5*(index//10)) == 0:
        return 0.01
    else:
        return 0.01/(5*(index//10))

learn_callback = keras.callbacks.LearningRateScheduler(learningRateScheduler)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=1540/32,
    epochs=30,
    max_queue_size=30,
    workers=8,
    validation_data=test_generator,
    validation_steps=200,
    callbacks=[learn_callback]
)


score = model.evaluate_generator(test_generator,
                                steps=200,
                                workers=8,
                                max_queue_size=10,
                                )



print(model.metrics_names)
print(score)

print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()