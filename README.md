---
# The Repository is ARCHIVED!
### it is now maintained in https://codebase.helmholtz.cloud/m-team/ai/ai4os-fasterrcnn-torch <br>(mirrored to https://github.com/ai4os-hub/ai4os-fasterrcnn-torch)
---

# fasterrcnn_pytorch_api

[![Build Status](https://jenkins.indigo-datacloud.eu/buildStatus/icon?job=Pipeline-as-code/DEEP-OC-org/fasterrcnn_pytorch_api/master)](https://jenkins.indigo-datacloud.eu/job/Pipeline-as-code/job/DEEP-OC-org/job/fasterrcnn_pytorch_api/job/master)

[This external repository](https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline)  provides a pipeline for training PyTorch FasterRCNN models on custom datasets. With this pipeline, you have the flexibility to choose between official PyTorch models trained on the COCO dataset, use any backbone from Torchvision classification models, or even define your own custom backbones. The trained models can be used for object detection tasks on your specific datasets.
## Training and Inference

To learn how to train and perform inference using this pipeline, please refer to the following links:
- [How to Train Faster RCNN ResNet50 FPN V2 on Custom Dataset?](https://debuggercafe.com/how-to-train-faster-rcnn-resnet50-fpn-v2-on-custom-dataset/#download-code): This tutorial provides a step-by-step guide on training the Faster RCNN model with the ResNet50 FPN V2 backbone on your custom dataset.
- [Small Scale Traffic Light Detection using PyTorch](https://debuggercafe.com/small-scale-traffic-light-detection/): This article demonstrates a specific use case where the pipeline is used for detecting traffic lights in small-scale images.

More information about the model can be found in the [external repository](https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline). You can explore additional details and documentation provided in that repository.

## Adding DeepaaS API into the existing codebase
In this repository, we have integrated a DeepaaS API into the existing codebase, enabling the seamless utilization of this pipeline. The inclusion of the DeepaaS API enhances the functionality and accessibility of the code, making it easier for users to leverage and interact with the pipeline efficiently.

><span style="color:Blue">**Note:**</span> To streamline the process of integrating the API into the external repository and eliminate code duplication, we decided to include a fork of the external repository as a submodule within the API repository. This approach allows us to maintain a separate repository for the API while still leveraging the shared codebase from the external repository, ensuring efficient collaboration and updates between the two projects.



## Install the API and the external submodule requirement

To launch the API, first, install the package, and then run [DeepaaS](https://github.com/indigo-dc/DEEPaaS):

```bash
git clone --depth 1 https://codebase.helmholtz.cloud/m-team/ai/fasterrcnn_pytorch_api.git
cd fasterrcnn_pytorch_api
git submodule init
git submodule update --remote --merge
pip install -e ./path/to/submodule/dir
pip install -e .
```
><span style="color:Blue">**Note:**</span> Before installing the API and submodule requirements, please make sure to install the following system packages: `gcc`, `unzip`, and `libgl1` as well. These packages are essential for a smooth installation process and proper functioning of the framework.
```
apt update
apt install -y unzip
apt install -y gcc
apt install -y libgl1
apt install -y libglib2.0-0
```

><span style="color:Blue">**Note:**</span>  The associated Docker container for this module can be found at: https://codebase.helmholtz.cloud/m-team/ai/DEEP-OC-fasterrcnn_pytorch_api.git

## Project Structure

```
├── LICENSE                <- License file
│
├── README.md              <- The top-level README for developers using this project.
│
├── requirements.txt       <- The requirements file for reproducing the analysis environment, e.g., generated with `pip freeze > requirements.txt`
│
├── setup.py, setup.cfg    <- Makes the project pip installable (`pip install -e .`) so that fasterrcnn_pytorch_api can be imported
│
├── fasterrcnn_pytorch_api <- Source code for use in this project.
│   ├── config            <- API configuration subpackage
│   ├── scripts           <- API scripts subpackage for predictions and training the model
│   ├── __init__.py       <- File for initializing the python library
│   ├── api.py            <- API core module for endpoint methods
│   ├── fields.py         <- API core fields for arguments
│   └── utils_api.py      <- API utilities module
│
├── Jenkinsfile           <- Describes the basic Jenkins CI/CD pipeline
├── data                  <- Folder to store data for training and prediction
└── models                <- Folder to store checkpoints
```

## Dataset Preparation

To train the FasterRCNN model, your annotations should be saved as Pascal VOC XML formats. Please organize your data in the following structure:
```
data
│
└── my_dataset
    ├── train_imgs
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   ├── ...
    ├── train_labels
    │   ├── img1.xml
    │   ├── img2.xml
    │   ├── ...
    ├── valid_imgs
    │   ├── img_1.jpg
    │   ├── img_2.jpg
    │   ├── ...
    ├── valid_labels
    │   ├── img_1.xml
    │   ├── img_2.xml
    │   ├── ...
    └── config.yaml

```

The `config.yaml` file contains the following information about the data:

```yaml
# Images and labels directory should be insade 'fasterrcnn_pytorch_api/data' directory.
TRAIN_DIR_IMAGES: 'my_dataset/train_imgs'
TRAIN_DIR_LABELS: 'my_dataset/train_labels'
VALID_DIR_IMAGES: 'my_dataset/valid_imgs'
VALID_DIR_LABELS: 'my_dataset/valid_labels'
# Class names.
CLASSES: [
    class1, class2, ...
]
# Number of classes.
NC: n
```
## Available Backbones for FasterRCNN

The following backbones are available for training:


``` 
    'fasterrcnn_convnext_small'
    'fasterrcnn_convnext_tiny',
    'fasterrcnn_efficientnet_b0',
    'fasterrcnn_efficientnet_b4',
    'fasterrcnn_mbv3_small_nano_head',
    'fasterrcnn_mbv3_large',
    'fasterrcnn_mini_squeezenet1_1_small_head',
    'fasterrcnn_mini_squeezenet1_1_tiny_head',
    'fasterrcnn_mobilenetv3_large_320_fpn',
    'fasterrcnn_mobilenetv3_large_fpn',
    'fasterrcnn_nano',
    'fasterrcnn_resnet18',
    'fasterrcnn_resnet50_fpn_v2',
    'fasterrcnn_resnet50_fpn',
    'fasterrcnn_resnet101',
    'fasterrcnn_resnet152',
    'fasterrcnn_squeezenet1_0',
    'fasterrcnn_squeezenet1_1_small_head',
    'fasterrcnn_squeezenet1_1',
    'fasterrcnn_vitdet',
    'fasterrcnn_vitdet_tiny',
    'fasterrcnn_mobilevit_xxs',
    'fasterrcnn_regnet_y_400mf'

```
## Augmentation Options

During the training of the model, you have the following augmentation options available:


- **Blur**: Apply blurring effect with a probability of 10% (p=0.1) and a maximum blur kernel size of 3 (blur_limit=3).
- **Motion Blur**: Apply motion blurring with a probability of 10% (p=0.1) and a maximum blur kernel size of 3 (blur_limit=3).
- **Median Blur**: Apply median blurring with a probability of 10% (p=0.1) and a maximum blur kernel size of 3 (blur_limit=3).
- **To Gray**: Convert the image to grayscale with a probability of 10% (p=0.1).
- **Random Brightness and Contrast**: Apply random changes in brightness and contrast with a probability of 10% (p=0.1).
- **Color Jitter**: Perform color jittering with a probability of 10% (p=0.1).
- **Random Gamma**: Apply random gamma correction with a probability of 10% (p=0.1).
- **Horizontal Flip**: Perform horizontal flipping with a probability of 100% (p=1.0).
- **Vertical Flip**: Perform vertical flipping with a probability of 100% (p=1.0).
- **Rotation**: Rotate the image within a range of -45 to 45 degrees (limit=45).
- **Shift, Scale, and Rotate**: Perform combined shift, scale, and rotation transformation with a probability of 0% (p=0.0), shift limit of 0.1, scale limit of 0.1, and rotation limit of 30 degrees.
- **Cutout**: Apply cutout augmentation with no holes (num_holes=0), a maximum hole height of 0 (max_h_size=0), a maximum hole width of 8 (max_w_size=8), and fill with zeros, using a probability of 0% (p=0.0).
- **Channel Shuffle**: Shuffle color channels with a probability of 0% (p=0.0).

Remember that you can control the usage of these options by adjusting the `p`(probability) value. If you want to exclude a particular augmentation option, simply set its `p` value to 0.

Feel free to customize these options based on your dataset characteristics and project requirements to achieve the best results.
## Launching the API

To train the model, run:
```
deepaas-run --listen-ip 0.0.0.0
```
Then, open the Swagger interface, change the hyperparameters in the train section, and click on train.

><span style="color:Blue">**Note:**</span>  Please note that the model training process may take some time depending on the size of your dataset and the complexity of your custom backbone. Once the model is trained, you can use the API to perform inference on new images.

## Using wandb to track your experience

Weights & Biases ([wandb](https://wandb.ai/)) simplifies the process of tracking your experiments, managing and versioning your data, and fostering collaboration within your team. With these tasks taken care of, you can direct your full attention to building the most optimal models. If you want to use [wandb](https://wandb.ai/) to track your experience, make sure to follow these steps:
1. Sign up in [here](https://wandb.ai/).
2. Copy Your API key for logging in to the wandb library.
3. change the value of disable_wandb in the training arguments to False.
4. In the `./fasterrcnn_pytorch_api/fasterrcnn_pytorch_api/configs/setting.ini` file, set the following parameter
```
[wandb_token]
#your token to save and monitor your training model in the wandb
token= your_token #replace with your actual token string
```
5. To track your experiments, simply log in [here](https://wandb.ai/).

## Inference Methods

You can utilize the Swagger interface to upload your images or videos and obtain the following outputs:

- For images:

    - An annotated image highlighting the object of interest with a bounding box.
    - A JSON string providing the coordinates of the bounding box, the object's name within the box, and the confidence score of the object detection.

- For videos:

    - A video with bounding boxes delineating objects of interest throughout.
    - A JSON string accompanying each frame, supplying bounding box coordinates, object names within the boxes, and confidence scores for the detected objects.

# Use rclone to copy from remote

If you want to use rclone to download trained models from nextcloud
set  `use_rclone` in the `./fasterrcnn_pytorch_api/fasterrcnn_pytorch_api/configs/setting.ini` to true and pass the path to your model directory on the nextcloud
in the remote variable. 
```
[use_rclone]
value= False 

[remote]
# Directory containing trained model for prediction  on the nextcloud  
path = models_sub
```
If you have not already configured the rclone in your env, you can configure your rclone credentials in the same script otherwise leave it as it is.

```
[RCLONE_CONFIG_RSHARE_PASS]
#Rclone password
password= <password> # replce with your  password
[RCLONE_CONFIG_RSHARE_USER]
#Rclone username
username= user_name # replce with your username
[RCLONE_CONFIG_RSHARE_TYPE]
#Rclone type
type= webdav
[RCLONE_CONFIG_RSHARE_URL]
#Rclone url
url= DEEP_IAM-da6568e7-7fad-43ac-a124-b0f58994d988
[RCLONE_CONFIG_RSHARE_VENDOR]
#Rclone vendor
vendor= ''
[RCLONE_CONFIG]
#Rclone path config
rclone_config = /srv/.rclone/rclone.conf #create this file on your machine and pu the path here

```
Give the timestamp of the model as a prediction argument and use it for inference purposes.
