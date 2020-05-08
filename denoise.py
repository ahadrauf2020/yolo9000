import os
import cv2
import glob
import shutil
from os import path

# Edit
main_repository_path = ""

train = "train"
val = "val"
test = "test"

train_folder = train + "/"
new_train_folder = train + "_adversarial/"
val_folder = val + "/"
test_folder = test + "/"

images_folder = "images/"

old_folder_name = "tiny-imagenet-200/"
new_folder_name = "tiny-imagenet-200/"

path_to_data = "" + old_folder_name

JPEG = ".JPEG"

names_of_directories = [name for name in os.listdir(main_repository_path + path_to_data + train_folder)]

if not path.exists(main_repository_path + new_folder_name):
    os.mkdir(main_repository_path + new_folder_name)
    os.mkdir(main_repository_path + new_folder_name + train_folder)

    shutil.copy(main_repository_path + path_to_data + "wnids.txt", main_repository_path + new_folder_name)
    shutil.copy(main_repository_path + path_to_data + "words.txt", main_repository_path + new_folder_name)

for directory_name in names_of_directories:
    directory_name_folder_path = main_repository_path + new_folder_name + new_train_folder + directory_name
    if path.exists(directory_name_folder_path):
        continue

    txt_file_name = directory_name + "_boxes.txt"
    os.mkdir(directory_name_folder_path)
    shutil.copy(main_repository_path + path_to_data + train_folder + directory_name + "/" + txt_file_name, main_repository_path + new_folder_name + new_train_folder + directory_name + "/" + txt_file_name)
    directory_name_folder_images_path = main_repository_path + new_folder_name + new_train_folder + directory_name + "/" + images_folder
    os.mkdir(directory_name_folder_images_path)
    for img_number in range(0, 500):
        image_name = str(directory_name) + "_" + str(img_number) + JPEG
        img = cv2.imread(main_repository_path + path_to_data + new_train_folder + str(directory_name) + "/" + images_folder + image_name) 
        denoised_img = cv2.fastNlMeansDenoisingColored(img, None, 10, 7, 21) 
        img_dir = directory_name_folder_images_path + "/"
        cv2.imwrite(img_dir + image_name, denoised_img)
    print(directory_name, 'Finished')
print("done")
    