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

The associated Docker container for this module can be found at: https://git.scc.kit.edu/m-team/ai/DEEP-OC-fasterrcnn_pytorch_api.git

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

To train the FasterRCNN model, your dataset should be in coco format (.xml). Organize your data in the following structure:
```
data
    ├── train_imgs
    │   ├── img1.jpg
    │   ├── img2.jpg
    ├── train_labels
    │   ├── img1.xlm
    │   ├── img2.xml
    ├── valid_imgs
    │   ├── img3.jpg
    │   ├── img4.jpg
    ├── valid_labels
    │   ├── img1.xlm
    │   ├── img2.xml
    └── config.yaml

```

The `config.yaml` file contains the following information about the data:

```yaml
# Images and labels directory should be relative to train.py
TRAIN_DIR_IMAGES: '../data/train_imgs'
TRAIN_DIR_LABELS: '../data/train_labels'
VALID_DIR_IMAGES: '../data/valid_imgs'
VALID_DIR_LABELS: '../data/valid_labels'
# Class names.
CLASSES: [
    class1, class2, ...
]
# Number of classes.
NC: 2
```

## Launching the API

To train the model, run:
```
deepaas-run --listen-ip 0.0.0.0
```
Then, open the Swagger interface, change the hyperparameters in the train section, and click on train.

Note1: Please note that the model training process may take some time depending on the size of your dataset and the complexity of your custom backbone. Once the model is trained, you can use the API to perform object detection on new images.

Note2: If you want to use Wandb to track your experience, make sure to follow these steps:
1. Change the value of `disable_wandb` to `False`.
2. In the `./fasterrcnn_pytorch_api/fasterrcnn_pytorch_api/configs/setting.ini` file, set the `wandb_token` to your actual Wandb API token. 

