# fasterrcnn_pytorch_api

[![Build Status](https://jenkins.indigo-datacloud.eu/buildStatus/icon?job=Pipeline-as-code/DEEP-OC-org/UC--fasterrcnn_pytorch_api/master)](https://jenkins.indigo-datacloud.eu/job/Pipeline-as-code/job/DEEP-OC-org/job/UC--fasterrcnn_pytorch_api/job/master)

[This external repository](https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline)  provides a pipeline for training PyTorch FasterRCNN models on custom datasets. With this pipeline, you have the flexibility to choose between official PyTorch models trained on the COCO dataset, use any backbone from Torchvision classification models, or even define your own custom backbones. The trained models can be used for object detection tasks on your specific datasets.
## Training and Inference

To learn how to train and perform inference using this pipeline, please refer to the following links:
- [How to Train Faster RCNN ResNet50 FPN V2 on Custom Dataset?](https://debuggercafe.com/how-to-train-faster-rcnn-resnet50-fpn-v2-on-custom-dataset/#download-code): This tutorial provides a step-by-step guide on training the Faster RCNN model with the ResNet50 FPN V2 backbone on your custom dataset.
- [Small Scale Traffic Light Detection using PyTorch](https://debuggercafe.com/small-scale-traffic-light-detection/): This article demonstrates a specific use case where the pipeline is used for detecting traffic lights in small-scale images.

More information about the model can be found in the [external repository](https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline). You can explore additional details and documentation provided in that repository.

In the current repository, we have integrated a DeepaaS API for this existing pipeline, making it easy to access and use the trained models for inference.

## Install the API and the external submodule requirement

To launch the API, first, install the package, and then run [DeepaaS](https://github.com/indigo-dc/DEEPaaS):

```bash
git clone --depth 1 https://github.com/falibabaei/fasterrcnn_pytorch_api
cd fasterrcnn_pytorch_api
git submodule init
git submodule update
pip install -e ./path/to/submodule/dir
pip install -e .
```
><span style="color:Blue">**Note:**</span> Before installing the API and submodule requirements, please make sure to install the following system packages: `gcc`, `unzip`, and `libgl1` as well. These packages are essential for a smooth installation process and proper functioning of the framework.
```
apt update
apt install -y unzip
apt install -y gcc
apt install -y libgl1
```

><span style="color:Blue">**Note:**</span>  The associated Docker container for this module can be found at: https://git.scc.kit.edu/m-team/ai/DEEP-OC-fasterrcnn_pytorch_api.git

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

To train the FasterRCNN model, your annotations should be saved as XML files. Please organize your data in the following structure:
```
data
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
TRAIN_DIR_IMAGES: '../my_dataset/train_imgs'
TRAIN_DIR_LABELS: '../my_dataset/train_labels'
VALID_DIR_IMAGES: '../my_dataset/valid_imgs'
VALID_DIR_LABELS: '../my_dataset/valid_labels'
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
5. To track your experiments, simply log in here.

