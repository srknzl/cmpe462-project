from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.models import Sequential
from keras.utils import to_categorical, plot_model
from keras.optimizers import SGD
import cv2
import subprocess 
import os 
import numpy as np
import matplotlib.pyplot as plt
import imgaug as ia
from imgaug import augmenters as iaa

ia.seed(4)

BATCH_SIZE = 32
EPOCHS = 5


animal_folders_list = [
    'bear',
    'cougar',
    'cow',
    'coyote',
    'deer',
    'elephant',
    'giraffe',
    'goat',
    'gorilla',
    'horse',
    'kangaroo',
    'leopard',
    'lion',
    'panda',
    'penquin',
    'sheep',
    'skunk', 
    'tiger',
    'zebra']
classname_to_class_number = {
    'bear' : 0,
    'cougar' : 1,
    'cow' : 2,
    'coyote': 3,
    'deer': 4,
    'elephant' : 5 ,
    'giraffe' : 6,
    'goat' : 7,
    'gorilla' : 8,
    'horse' : 9,
    'kangaroo' : 10,
    'leopard' : 11,
    'lion' : 12,
    'panda' : 13,
    'penquin' : 14,
    'sheep' : 15,
    'skunk' : 16, 
    'tiger' : 17,
    'zebra' : 18
}

class_number_to_classname = {
    0 :'bear',
    1 :'cougar',
    2 :'cow',
    3 :'coyote',
    4 :'deer',
    5 :'elephant',
    6 :'giraffe',
    7 :'goat',
    8 :'gorilla',
    9 :'horse',
    10 :'kangaroo',
    11 :'leopard',
    12 :'lion',
    13 :'panda',
    14 :'penquin',
    15 :'sheep',
    16 :'skunk', 
    17 :'tiger',
    18 :'zebra'
}
animal_images = {} # keys: animal names, values: array of images 

for animal_name in animal_folders_list:  # remove trailing whitespaces after animal names
    animal_images[animal_name] = []

for animal_folder_name in animal_folders_list: # find the file names in the folders and read images, store them in animal_images object
    ls_output = subprocess.check_output(["ls","animal_database/"+ animal_folder_name +"/original"])
    files = ls_output.decode().split("\n") # decode is needed because output is in b'some string' format. converts into -> 'some string'
    for x in range(len(files)): # remove if there is empty
        if files[x]=='':
            del files[x]
    for file in files: # read and store in animal_images
        animal_images[animal_folder_name].append(cv2.imread("animal_database/"+ animal_folder_name + "/original/" + file)[:, :, ::-1])

print("------------DATASET SUMMARY-------------")
for animal in animal_images: # print dataset summary
    print(animal+" : "+str(len(animal_images[animal])))

for animal in animal_images: # preprocessing: resize images as (300,300,3)
    for x in range(len(animal_images[animal])):
        animal_images[animal][x] = cv2.resize(animal_images[animal][x],(300,300))


for animal in animal_images: 
    gaus = iaa.AdditiveGaussianNoise(scale=(10, 60))
    crop = iaa.Crop(percent=(0, 0.2))
    rotate = iaa.Affine(rotate=(-25, 25))

    rot_images_aug = np.array(rotate.augment_images(animal_images[animal]))
    animal_images[animal] = np.concatenate((animal_images[animal], rot_images_aug))

    gaus_images_aug = np.array(gaus.augment_images(animal_images[animal]))
    animal_images[animal] = np.concatenate((animal_images[animal], gaus_images_aug))

    crop_images_aug = np.array(crop.augment_images(animal_images[animal]))
    animal_images[animal] = np.concatenate((animal_images[animal], crop_images_aug))


for animal in animal_images: # changing images to 0 1 float format  
    for x in range(len(animal_images[animal])):
        animal_images[animal][x] = animal_images[animal][x]/255.

print("------------DATASET SUMMARY AFTER AUGMENTATION -------------")
for animal in animal_images: # print dataset summary
    print(animal+" : "+str(len(animal_images[animal])))

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cmpe462_kth_animals.h5'

training_set = {}
validation_set = {}
test_set = {}

# splitting the dataset into training, validation and test set with 80,10,10 ratio.
# first find 10 percent(int division), allocate that amount to test and val sets from beginning of the dataset
# and then the rest is training set.
for animal in animal_images:
    ten_percent = len(animal_images[animal])//10
    test_set[animal] = animal_images[animal][0:ten_percent]
    validation_set[animal] = animal_images[animal][ten_percent+1:ten_percent*2+1]
    training_set[animal] = animal_images[animal][ten_percent*2+1:]

training_set_as_array = []
training_labels_as_array = []

for classname in training_set:
    for animal in training_set[classname]:
        training_set_as_array.append(animal)
        training_labels_as_array.append(classname_to_class_number[classname])

validation_set_as_array = []
validation_labels_as_array = []

for classname in validation_set:
    for animal in validation_set[classname]:
        validation_set_as_array.append(animal)
        validation_labels_as_array.append(classname_to_class_number[classname])

test_set_as_array = []
test_labels_as_array = []

for classname in test_set:
    for animal in test_set[classname]:
        test_set_as_array.append(animal)
        test_labels_as_array.append(classname_to_class_number[classname])

# Convert to numpy array for keras 
training_set_as_array  = np.array(training_set_as_array)
training_labels_as_array  = np.array(training_labels_as_array)
validation_set_as_array  = np.array(validation_set_as_array)
validation_labels_as_array  = np.array(validation_labels_as_array)
test_set_as_array  = np.array(test_set_as_array)
test_labels_as_array  = np.array(test_labels_as_array)


# Convert to one hot 

training_labels_as_array = to_categorical(training_labels_as_array,19,"int32")
validation_labels_as_array = to_categorical(validation_labels_as_array,19,"int32")
test_labels_as_array = to_categorical(test_labels_as_array,19,"int32")

model = Sequential()
model.add(Conv2D(32, (3,3) , input_shape=(300,300,3)))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (3,3)))
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
sgd = SGD(lr=0.01)

model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy']
)
plot_model(model, to_file='model.png',show_shapes=True)



history = model.fit(training_set_as_array,
                    training_labels_as_array,
                    shuffle=True,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(validation_set_as_array,validation_labels_as_array)
)
model.save_weights(save_dir+ "/" +model_name)

score = model.evaluate(test_set_as_array, test_labels_as_array)
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
