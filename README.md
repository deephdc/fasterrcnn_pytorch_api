# MLFlow_fasterrcnn

## Getting started

Object detection using FasterRCNN model
Introduce MLFlow as a solution:
    * to track all experiments during model training (see mlflow_train.py code) 
    * store the best trained model in the Model Registry (see mlflow_train.py code) 
    * serve the model (after deployment) to be consumed by users for prediction purposes on new data. (see mlflow_inference.py code) 

To launch it, first install the package then run [deepaas](https://github.com/indigo-dc/DEEPaaS):
```bash
git clone https://github.com/falibabaei//fasterrcnn_pytorch_api
cd fasterrcnn_pytorch_api
pip install -e .
git submodule init
git submodule update
cd submodule/dir
pip install -e .
cd ..

```
The associated Docker container for this module can be found in https://github.com/falibabaei//DEEP-OC-fasterrcnn_pytorch_api.

You might need to install some other dependent packages/libraries as follows:
```bash 
pip install setuptools wheel opencv-python gcc libgl1  vision_transformers albumentations
```

### Prerequisites
** Main installation **
you have:
<ul>
<li>python3 > 3.7 but lower than 3.11
<li>pip is installed

</ul>

### Install mlflow framework
pip install mlflow[extras]


## Configuring MLFlow constants

**Remote MLFlow server**
MLFLOW_REMOTE_SERVER="http://mlflow.dev.ai4eosc.eu" <your mlflow_tracking_server>
#Set the MLflow server and backend and artifact stores
mlflow.set_tracking_uri(MLFLOW_REMOTE_SERVER)

** for direct API calls via HTTP we need to inject credentials**
MLFLOW_TRACKING_USERNAME = 'mlflow_user'
MLFLOW_TRACKING_PASSWORD =  getpass.getpass()  # inject password by typing manually

**for MLFLow-way we have to set the following environment variables**
os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD

## Acknowledgment
This work is co-funded by AI4EOSC project that has received funding from the European Union's Horizon Europe 2022 research and innovation programme under agreement No 101058593

## License
For open source projects, say how it is licensed.



