import keras
import numpy as np
import cv2

# Data creation, splitting into test and training set 



model = keras.models.Sequential()
model.add(keras.layers.Dense(units=64, activation='relu', input_dim=100))
model.add(keras.layers.Dense(units=64, activation='relu', input_dim=100))
model.add(keras.layers.Dense(units=10, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True),
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=32)
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

classes = model.predict(x_test, batch_size=128)

