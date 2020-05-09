from collections import OrderedDict
import shutil 
import os

# Move images so that dataloader can handle it
class_maps = OrderedDict()
with open('./data/tiny-imagenet-200/val/val_annotations.txt', "r") as f:
    for line in f:
        one_line = line.split()
        class_maps[one_line[0]] = one_line[1]
# print(class_maps)


val_path = './data/tiny-imagenet-200/val/'
for image, class_name in class_maps.items():
    if not os.path.exists(val_path+'{}'.format(class_name)):
        try:
            os.makedirs(val_path+'{}/images'.format(class_name))
        except OSError as e:
            raise e
    if os.path.exists(val_path+'images'):
        source = val_path + 'images/{}'.format(image)
        destination = val_path + '{}/images/'.format(class_name)
        shutil.move(source, destination)

# This line should remove the original directory if we moved all images properly
os.rmdir(val_path + 'images')