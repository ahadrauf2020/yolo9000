# Robust and Generalizable Computer Vision Classification
<b>Course:</b> CS182 (https://bcourses.berkeley.edu/courses/1487769/pages/cs-l-w-182-slash-282a-designing-visualizing-and-understanding-deep-neural-networks-spring-2020)

<b>Group Name:</b> YOLO9000

<b>Authors:</b> Ahad Rauf, Chris Sun, Michael Lavva, Kei Watanabe

<b>Link to the Paper:</b> [URL]

## Abstract:
We created a robust computer vision classifierthat performs well in a dataset that contains per-turbations. To achieve this, we used various dataaugmentation and other deep learning model tech-niques, such as model ensembling, denoising, ad-versarial training, and attention networks. Thesemethods helped improve the robustness againstboth naturally perturbed and adversarial datasets.In addition, we implemented an explainable AIcomponent to understand how the model makesits classification decisions.  Through these tech-niques, the goal was to improve the accuracy onan adversarial dataset.

<img src="images/class_action_maps.png" width="500">

## Setup / Dependencies:
This project requires PyTorch, OpenCV, NumPy, MatPlotLib, PIL, etc.

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
