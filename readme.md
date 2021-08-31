## General

    This project was devloped from the University of Hamburg. 
    08.2021
    You can also find it on [GitHub](https://github.com/Yoshi-E/Object-Localizer-Project)

## Goal

    The goals of this project are the following:
    - Become familar with the propular framework Tensorflow
    - Implement an easy to use framework to quickly use different models & compare them
    - Train different neural networks to detect objects (i.e. Robots)
    - Compare the performance of said models for the purpose of real time application. 

## Requirments

    Tested on Python 3.6.8
    My Anaconda enviroment can be found in requirments.txt

    However generally `tensoreflow >= 2.6.0` and any `numpy` and `cv2` version should be sufficient

    Pretrained weights and the dataset can be downloaded below.

## Dataset

    The dataset for this project has been recorded and labled by myself and can be freely used to train and test neural networks.
    The file is ~1GB and contains 14.179 labled images, aswell as multiple training and validation combinations.

    [Download](https://eric.bergter.com/files/10_rosbag.zip)

### Dataset Details
    After the dataset has been downloaded, it should be placed in `FastDetector/datasets`.
    For example the path to an image should look like this: `FastDetector/datasets/10_rosbag/images/1565608338944067175.jpg`
    
    In the root of the dataset you can find the indexed files `train.csv` and `validation.csv`. These files contain the images and their bounding boxes that are used for training.
    The bounding boxes come in the format: x0, y0, x1, y1, class
    The class indicates the direction the robot is facing, and is not used in this project.

## Weights

    Trained weights can be downloaded here:
    Size: 335mb
    [Download](https://eric.bergter.com/files/weights.zip)
    Extract the weights to the root directory, so that e.g. `weights/MobiNetV2/front_weight-0.12` is valid path

## Models

    For the models I mostly choose pretrained networks, such as MobiNet or VGG16.
    MobiNet in peticular is a great network to work with, as its fast and easy to train. This could also be confirmed by my results.

## Metric

    All networks are trained to maximize the IOU (Intersection over Union) of the predicition and ground truth value.

    ![IOU](examples/Intersection-over-Union-IOU-calculation-diagram.png)

    This method is implemented in `models/MobiNetV2/validation.py`

## Evaluation

    For the evaluation all the models were tested on a single CPU core, with no GPU acceleration.
    The speed was mesured on the first 10.000 images of the training dataset with a resolution of 640x480px.
    The inital loading time of the models is not mesured, but the loading and preprocssing of every image is taken into account to simulate possible real conditions.

## Results

    - MobiNetV2 : TOTAL: 402.51s LOW: 0.03765 HIGH: 0.15395 ( 2,257,984 parameter)
    - MobiNetV3 : TOTAL: 376.01s LOW: 0.03296 HIGH: 0.16095 ( 1,529,968 parameter)
    - VGG16     : TOTAL: 588.61s LOW: 0.05225 HIGH: 0.19111 (14,714,688 parameter)


## Reproducing the results

    To train the networks simply run the respecive scripts in the folder FastDetector.
    The created weight files will be saved in the current working directory.

    The same way its also possible to test saved models on the dataset. Simply specify the weight and dataset in the script.

    The script `Test_Model_Performance.py` will load all the respective `*_test.py` script and will evaluate the performance for each model. 

## System

    The model was tested and trained on a Windows 10 PC:
    - CPU: Intel® Core™ i7-6800K Processor
    - GPU: NVIDIA GTX 1080   
    - Drive: SAMSUNG SSD 970 EVO PLUS 1TB

