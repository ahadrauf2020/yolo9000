<<<<<<< HEAD
# yolo9000
CS 182 Vision Project - Generalizable Classifiers

The link to download the model files:
https://drive.google.com/drive/folders/1Mxigts_aeI6LOo2BTjP7IfbCBAv4FCyV?usp=sharing
=======
# Robust and Generalizable Computer Vision Classification
<b>Course:</b> CS182 (https://bcourses.berkeley.edu/courses/1487769/pages/cs-l-w-182-slash-282a-designing-visualizing-and-understanding-deep-neural-networks-spring-2020)

<b>Group Name:</b> YOLO9000

<b>Authors:</b> Ahad Rauf, Chris Sun, Michael Lavva, Kei Watanabe

<b>Link to the Paper:</b> [URL]

## Abstract:
For the final project of CS182, we created a robust computer vision classifier that performs well in a dataset that contains perturbations. To achieve this, we used various data augmentation and other deep learning model techniques, such as model ensembling, denoising, and adversarial training. These  methods  helped  improve  the  robustness against both naturally perturbed and adversarial datasets. In addition, we implemented an explainable AI component to understand how the model makes its classification decisions.

<img src="images/class_action_maps.png" width="500">

## Setup / Dependencies:
This project requires PyTorch, OpenCV, NumPy, MatPlotLib, PIL, etc.

## How to run test_submission.py  
1. Create a conda environment and install the packages:  
conda create -n yolo9000-testing python=3.6 pip  
conda activate yolo9000-testing  
pip install -r requirements.txt  

2. Place data set such that relative path becomes ./data/tiny-imagenet-200/train/

3. Create a directory called models at the top directory and download all models from the google drive folder and place them inside models directory : https://drive.google.com/drive/folders/1LVNFqUJAmhSGgYT4cVlzuImfZKi2ctyI?usp=sharing    

5. Run `python test_submission_torch.py 'eval.csv'` where you need to replace 'eval.csv' with the path the eval.csv  


## Dataset:
The dataset for this project is from https://tiny-imagenet.herokuapp.com/. To download it, go to the data folder in the master branch and run `get_data.sh`.

## Project Work:
There are several branches that include the work in the project:
* master: Contains denoising script.
* dev-ahad-adversarial: Generates adversarial examples.
* dev-ahad-finite-horizons
* dev-ahad-master
* dev-ahad-xai: Code for Explainable AI.
* dev-kei-resnet-vgg: Model & Snapshot Ensembling code
* resnet: Code for ResNet model.
* test-submission: Test submission code.
>>>>>>> 781060c84d2cd70d2df9b8e2244cc8cf72f00f98
