# CS 182 Vision Project - Generalizable Classifiers
<b>Group Name:</b> Yolo9000

<b>Link to the Paper:</b> [URL]

## Abstract:
For the final project of CS182, we created a robustcomputer vision classifier that performs well ina dataset that contains perturbations. To achievethis, we used various data augmentation and otherdeep learning model techniques, such as modelensembling, denoising, and adversarial training.These  methods  helped  improve  the  robustnessagainst both naturally perturbed and adversarialdatasets. In addition, we implemented an explain-able AI component to understand how the modelmakes its classification decisions. Through thesetechniques, the goal was to improve the model’saccuracy in an adversarial dataset

## Setup:
This project requires PyTorch, OpenCV, NumPy, MatPlotLib, PIL, etc.

## Dataset:
The dataset for this project is from https://tiny-imagenet.herokuapp.com/. To download it, go to the data folder in the master branch and run get_data.sh.

## Project Work:
There are several branches that include the work in the project:
* master
* dev-ahad-adversarial
* dev-ahad-finite-horizons
* dev-ahad-master
* dev-ahad-xai
* dev-kei-resnet-vgg
* resnet
* test-submission
