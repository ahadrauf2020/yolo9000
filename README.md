# Robust and Generalizable Computer Vision Classification
<b>Course:</b> CS182 ([Homepage of the Course](https://bcourses.berkeley.edu/courses/1487769/pages/cs-l-w-182-slash-282a-designing-visualizing-and-understanding-deep-neural-networks-spring-2020))

<b>Group Name:</b> YOLO9000

<b>Authors:</b> Ahad Rauf, Chris Sun, Michael Lavva, Keisuke Watanabe

<b>Link to the Paper:</b> [PDF of the Paper](https://github.com/ahadrauf2020/yolo9000/blob/master/report/cs182_report_robust_and_generalizable_computer_vision_classification.pdf)

## Abstract:
We created a robust computer vision classifier that performs well in a dataset that contains perturbations. To achieve this, we used various data augmentation and other deep learning model techniques, such as model ensembling, image denoising, adversarial training, and attention networks. These methods helped improve the robustness against both naturally perturbed and adversarial datasets. In addition, we created class action map visualizations for our models to help understand how the model makes its classification decisions. Through these techniques, we achieved a 71.1% Top 1 Accuracy and 90.0% Top 5 Accuracy on the Tiny-ImageNet classification challenge.

<img src="images/class_action_maps.png" width="500">

## Setup / Dependencies:
This project requires PyTorch, OpenCV2, NumPy, MatPlotLib, PIL, etc.

## Environment:
The project's environment was a Google Cloud’s Deep Learning VM instance and Nvidia Tesla K80 GPU / Nvidia T4.

## Dataset:
The dataset for this project is from https://tiny-imagenet.herokuapp.com/. To download it, go to the data folder in the master branch and run `get_data.sh`.

## How to Run the Project
1. Create a conda environment and install the packages:
`conda create -n yolo9000-testing python=3.6 pip
conda activate yolo9000-testing
pip install -r requirements.txt`

2. Create a directory called data at the top directory and place data set such that relative path becomes ./data/tiny-imagenet-200/train/

3. Create a directory called models at the top directory and download all models from the google drive folder and place them inside models directory : https://drive.google.com/drive/folders/1LVNFqUJAmhSGgYT4cVlzuImfZKi2ctyI?usp=sharing

4. Run `python test_submission.py 'eval.csv'` where you need to replace 'eval.csv' with the path to the eval.csv

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
