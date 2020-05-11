# CS 182 Vision Project - Generalizable Classifiers
<b>Group Name:</b> Yolo9000

<b>Link to the Paper:</b> [URL]

## Abstract:
For the final project of CS182, we created a robust computer vision classifier that performs well in a dataset that contains perturbations. 

To achieve this, we used various data augmentation and other deep learning model techniques, such as model ensembling, denoising, and adversarial training. These  methods  helped  improve  the  robustness against both naturally perturbed and adversarial datasets. In addition, we implemented an explainable AI component to understand how the model makes its classification decisions. Through these techniques, the goal was to improve the model’s accuracy in an adversarial dataset.

## Setup:
This project requires PyTorch, OpenCV, NumPy, MatPlotLib, PIL, etc.

## Dataset:
The dataset for this project is from https://tiny-imagenet.herokuapp.com/. To download it, go to the data folder in the master branch and run `get_data.sh`.

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
