import keras
import cv2
import subprocess 

animal_folder_list_file = open("animal_database/animal_folders.txt","r")
animal_folders_list = animal_folder_list_file.readlines()
animal_images = {} # keys: animal names, values: array of images 

for x in range(len(animal_folders_list)):  # remove trailing whitespaces after animal names
    animal_folders_list[x] = animal_folders_list[x].strip()
    animal_images[animal_folders_list[x]] = []

for animal_folder_name in animal_folders_list: # find the file names in the folders and read images, store them in animal_images object
    ls_output = subprocess.check_output(["ls","animal_database/"+ animal_folder_name +"/original"])
    files = ls_output.decode().split("\n") # decode is needed because output is in b'some string' format. converts into -> 'some string'
    for x in range(len(files)): # remove if there is empty
        if files[x]=='':
            del files[x]
    for file in files: # read and store in animal_images
        animal_images[animal_folder_name].append(cv2.imread("animal_database/"+ animal_folder_name + "/original/" + file))

print("------------DATASET SUMMARY-------------")
for animal in animal_images: # print dataset summary
    print(animal+" : "+str(len(animal_images[animal])))


for animal in animal_images: # resize images as (300,300,3)
    for x in range(len(animal_images[animal])):
        animal_images[animal][x] = cv2.resize(animal_images[animal][x],(300,300))


# model = keras.models.Sequential()
# model.add(keras.layers.Dense(units=64, activation='relu', input_dim=100))
# model.add(keras.layers.Dense(units=64, activation='relu', input_dim=100))
# model.add(keras.layers.Dense(units=10, activation='softmax'))
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True),
#               metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=5, batch_size=32)
# loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

# classes = model.predict(x_test, batch_size=128)

