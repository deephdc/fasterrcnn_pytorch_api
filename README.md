# fasterrcnn_pytorch_api
[![Build Status](https://jenkins.indigo-datacloud.eu/buildStatus/icon?job=Pipeline-as-code/DEEP-OC-org/UC--fasterrcnn_pytorch_api/master)](https://jenkins.indigo-datacloud.eu/job/Pipeline-as-code/job/DEEP-OC-org/job/UC--fasterrcnn_pytorch_api/job/master)

Object detection using FasterRCNN model

To launch it, first install the package then run [deepaas](https://github.com/indigo-dc/DEEPaaS):
```bash
git clone https://github.com/falibabaei//fasterrcnn_pytorch_api
cd fasterrcnn_pytorch_api
pip install -e .
git submodule init
git submodule update
cd submodule/
pip install e .
deepaas-run --listen-ip 0.0.0.0
```
The associated Docker container for this module can be found in https://github.com/falibabaei//DEEP-OC-fasterrcnn_pytorch_api.

## Project structure
```
├── LICENSE                <- License file
│
├── README.md              <- The top-level README for developers using this project.
│
├── requirements.txt       <- The requirements file for reproducing the analysis environment, e.g.
│                             generated with `pip freeze > requirements.txt`
│
├── setup.py, setup.cfg    <- makes project pip installable (pip install -e .) so
│                             fasterrcnn_pytorch_api can be imported
│
├── fasterrcnn_pytorch_api    <- Source code for use in this project.
│       ├── config                  <- API configuration subpackage
│       ├── scripts                 <- API scripts subpackage for predictions
│       ├── __init__.py             <- File for initializing the python library
│       ├── api.py                  <- API core module for endpoint methods
│       ├── fields.py               <- API core fields for arguments
│       └── utils.py                <- API utilities module
└── Jenkinsfile   <- Describes basic Jenkins CI/CD pipeline
├── data                <- Folder to store data for training and prediction
├── models                  <- Folder to store checkpoints

```
