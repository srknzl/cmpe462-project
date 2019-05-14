import keras
import cv2
import subprocess 
import os 

animal_folders_list = ['bear', 'cougar', 'cow', 'coyote', 'deer', 'elephant', 'giraffe', 'goat', 'gorilla', 'horse', 'kangaroo', 'leopard', 'lion', 'panda', 'penquin', 'sheep', 'skunk', 'tiger', 'zebra']

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
        animal_images[animal_folder_name].append(cv2.imread("animal_database/"+ animal_folder_name + "/original/" + file))

print("------------DATASET SUMMARY-------------")
for animal in animal_images: # print dataset summary
    print(animal+" : "+str(len(animal_images[animal])))

for animal in animal_images: # resize images as (300,300,3)
    for x in range(len(animal_images[animal])):
        animal_images[animal][x] = cv2.resize(animal_images[animal][x],(300,300))

batch_size = 32
num_classes = 19
epochs = 100
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cmpe462_kth_animals.h5'
