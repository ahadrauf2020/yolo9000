This is some sample code for the CS 182/282 Computer Vision project (Tensorflow 2). It has the following files:



## How to run test_submission.py
1. Create a conda environment and install the packages:
conda create -n yolo9000-testing python=3.6 pip
conda activate yolo9000-testing
pip install -r requirements.txt

2. Create a directory called data at the top directory and place data set such that relative path becomes ./data/tiny-imagenet-200/train/

3. Create a directory called models at the top directory and download all models from the google drive folder and place them inside models directory : https://drive.google.com/drive/folders/1LVNFqUJAmhSGgYT4cVlzuImfZKi2ctyI?usp=sharing

4. Run `python test_submission.py 'eval.csv'` where you need to replace 'eval.csv' with the path to the eval.csv



README.txt - This file
requirements.txt - The python requirments necessary to run this project
train_sample.py - A sample training file which trains a simple model on the data, and save the checkpoint to be loaded
                  in the test_submission.py file.
test_submission.py - A sample file which will return an output for every input in the eval.csv
eval.csv - An example test file
data/get_data.sh - A script which will download the tiny-imagenet data into the data/tiny-imagenet-200 file

Note: You should be using Python 3 to run this code.
