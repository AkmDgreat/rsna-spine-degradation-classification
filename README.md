# RSNA 2024 Lumbar Spine Degenerative Classification

This Jupyter file is my submission for the [RSNA Spine Challenge](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification)

## How to run the notebook?

Method-1 (recommmended): Run it directly on Kaggle. You can find my notebook [here](https://www.kaggle.com/code/akmdgreat/kaggle-2024-spine-classification), with all the necessary datasets attached

Method-2: Run it locally. The original dataset is approximately 30 GB, but I processed it down to 4.5GB. You can download the dataset, and change the file paths, to run the
notebook locally.

## Overview of key functions

`load_dicom_stack()`: Processes a single MRI image into a Numpy array 

`create_dataset()`: Parses 30GB of MRI images to create a 4.5GB Numpy dataset. The function took 8 hours to run, and I saved the output, so you won't have to rerun the function

`create_tf_dataset()`:  Labels the dataset and formats it as a TensorFlow dataset, ready for model input.

`complete_training_workflow()`: creates and trains the model

`complete_testing_workflow()`: runs the trained model on the test set and converts the the model prediction (a Numpy array) to a `submission.csv` file

## Current plan

Implement image segmentation using [Medical SAM2](https://github.com/MedicineToken/Medical-SAM2)

Use the `DenseNet` architecture instead of the `EfficientNet`, as [this article](https://learnopencv.com/transfer-learning-for-medical-images/) suggests that older architectures perform better on MRI images

Explore domain adaptation techniques to improve the performance, because the images used in ImageNet are significantly different from MRI images

## Challenges faced

Initially, I tried to use the [DeepSpine](https://arxiv.org/abs/1807.10215) model architecture. Its a multi-input, multi-task, and multi-class model with SOTA performance. The model weights were not available online, so I decided to create and train the 50-million-parameter model from scratch. Unfortunately, I ran out of RAM while doing this.
    
Then, I decided to use transfer learning with a pre-trained EfficientNetV2 model. The problem? The model expects image of dimension `x * x * 3` while my images were of the dimension `x * x * 60 * 3`. An MRI is a stack of grey-scale images, just like a video is a stack of RGB images. Its unlike the images present in the `ImageNet` dataset. The articles [how to get started with deep learning using MRI data](https://medium.com/miccai-educational-initiative/how-to-get-started-with-deep-learning-using-mri-data-5d6a41dbc417) and [dealing with MRI and Deep Learning with Python](https://medium.com/towards-data-science/dealing-with-mri-and-deep-learning-with-python-c88f3dae0620) saved my life. 

The hardest part was figuring out how to process the images, labelling the dataset, and formatting it so that it can be fed into the model—all while keeping the memory usage under 30GB of RAM

## The result

After dedicating over two months to this project, I finally submitted my prediction. The result? I was ranked 954/1010. I am pleased to be in the 10%, nevermind it's the bottom 10% instead of the top