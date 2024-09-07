# Monitoring Motor Skills in Infant

This project features to monitor the motor skills in infants using Infant Pose Estimation.

## Introduction

Infant Pose Estimation is a computer vision project that detects and tracks the poses of infants. It provides valuable insights into their motor skills development. Currently, this project uses the [Infant-Pose-Estimation](https://github.com/ostadabbas/Infant-Pose-Estimation) repository, with additional enhancements and upgrades.

## Installation

To use this project, follow these steps:

1. Clone the [Infant-Pose-Estimation](https://github.com/ostadabbas/Infant-Pose-Estimation) repository.
2. Install the required dependencies as mentioned in the repository's README.
3. Copy the contents of the `Infant-Pose-Estimation` folder from this project into the cloned repository.


## Additional Scripts

Use these scripts to download the models and dataset. This also organizes them into specific folders.

```python
python Infant-Pose-Estimation/download_pretrained_fidip.py
```
This script is responsible for downloading and organizing the SyRIP dataset, which includes both annotations and images. The script performs the following tasks:

Download SyRIP Dataset: Fetches the SyRIP zip file from the specified URL.
Extract Dataset: Unzips the downloaded file into the data/syrip directory.
Organize Annotations: Moves and renames the annotation files to appropriate locations:
1000S/person_keypoints_train_pre_infant.json to annotations/person_keypoints_train_pre_infant.json
200R/person_keypoints_train_infant.json to annotations/person_keypoints_train_infant.json
validate100/person_keypoints_validate_infant.json to annotations/person_keypoints_validate_infant.json
Organize Images: Moves image files into corresponding directories:
SyRIP/images/train_pre_infant to images/train_pre_infant
SyRIP/images/train_infant to images/train_infant
SyRIP/images/validate_infant to images/validate_infant
This script ensures that the dataset is properly downloaded and structured for further processing.

```python
python Infant-Pose-Estimation/download_pretrained_hrnet.py
```
This script downloads a pre-trained HRNet model, which is essential for pose estimation tasks. The script follows these steps:

Define Download URL: Specifies the URL for the HRNet model file.
Create Directories: Ensures the existence of directories for storing the model file:
models/pytorch/imagenet
Download Model: Fetches the model file from the URL and saves it to the specified path:
models/pytorch/imagenet/hrnet_w48-8ef0771d.pth
By running this script, users can easily obtain the pre-trained HRNet model required for pose estimation.


```python
python Infant-Pose-Estimation/download_syrip_data.py
```
This script automates the download of multiple pre-trained models used for Fine-tuned Infant Detection and Pose Estimation (FIDIP). It includes the following functionalities:

Define Model URLs: Lists URLs for various pre-trained models:
hrnet_fidip.pth
mobile_fidip.pth
coco/posemobile.pth
Create Directories: Creates necessary directories to store the models:
models
models/coco
Download Models: Downloads each model file and saves it to the appropriate directory.
This script simplifies the process of obtaining multiple pre-trained models, ensuring they are organized and ready for use.

#### Run the model
- To run the model on a single picture use `Infant-Pose-Estimation/tools/predict_v6.py`.

This script performs pose estimation on a single image using a pre-trained model.

Load Model: Loads the pre-trained model for pose estimation.
Process Image: Reads and preprocesses the input image.
Predict Pose: Uses the model to predict pose keypoints.
Visualize Results: Draws the predicted keypoints on the image and saves the output.
This script is useful for applying pose estimation to individual images and visualizing the results.

- To run model on videos use `Infant-Pose-Estimation/tools/predict_video_v2.py`
  
This script performs pose estimation on a video file using a pre-trained model.

Load Model: Loads the pre-trained model for pose estimation.
Process Video: Reads the input video file frame by frame.
Predict Pose for Each Frame: Uses the model to predict pose keypoints for each frame.
Visualize and Save Results: Draws the predicted keypoints on each frame and saves the processed video.
This script is ideal for applying pose estimation to video files and generating visual outputs that show pose predictions for each frame.
