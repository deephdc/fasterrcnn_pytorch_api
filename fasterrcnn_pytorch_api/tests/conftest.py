"""Module for evolution requirements fixtures"""
# pylint: disable=redefined-outer-name

import fnmatch
import os
from pathlib import Path

import pytest
from deepaas.model.v2.wrapper import UploadedFile

from fasterrcnn_pytorch_api import configs, api, utils_api
 

DATA_FILES = os.listdir(configs.DATA_PATH)
MODELS_CKPT = utils_api.ls_local()
BACKBONES=configs.BACKBONES
 

@pytest.fixture(scope="module")
def metadata():
    """Fixture to return defined api metadata."""
    return api.get_metadata()
#####################################################
#  Fixtures to test train function
#####################################################

@pytest.fixture(scope="module", params=os.path.join(configs.DATA_PATH,'test_data/submarin.yaml'))#FIXME:name of the data config
def data_config(request):
    """Fixture to return data_configs argument to test."""
    return request.param

@pytest.fixture(scope="module", params=[None, os.path.join(configs.MODEL_DIR,'2023-05-10_121810/best_model.pth')]) #FIXME:path to model weights if using pretrained weights.
def weights(request):
    """Fixture to return  no label argument to test."""
    return request.param

@pytest.fixture(scope="module", params=configs.BACKBONES)
def model(request):
    """Fixture to return model checkpoint argument to test."""
    return request.param

@pytest.fixture(scope="module", params=[False, True])
def cosine_annealing(request):
    """Fixture to return cosine annealing argument to test."""
    return request.param

@pytest.fixture(scope="module", params=[False, True])
def resume_training(request):
    """Fixture to return resume training argument to test."""
    return request.param

@pytest.fixture(scope="module", params=[False, True])
def use_train_aug(request):
    """Fixture to return use train augmentations argument to test."""
    return request.param

@pytest.fixture(scope="module", params=[False, True])
def square_training(request):
    """Fixture to return square training argument to test."""
    return request.param

@pytest.fixture(scope="module", params=[False, True])
def no_mosaic(request):
    """Fixture to return no mosaic argument to test."""
    return request.param

@pytest.fixture(scope="module", params=[1])
def epochs(request):
    """Fixture to return number of epochs argument to test."""
    return request.param

@pytest.fixture(scope="module", params=[0])
def seed(request):
    """Fixture to return random seed argument to test."""
    return request.param

@pytest.fixture(scope="module", params=[4, 5])
def workers(request):
    """Fixture to return number of workers argument to test."""
    return request.param

@pytest.fixture(scope="module", params=[1])
def batch(request):
    """Fixture to return batch size argument to test."""
    return request.param

@pytest.fixture(scope="module", params=[0.001, 0.01])
def lr(request):
    """Fixture to return learning rate argument to test."""
    return request.param


@pytest.fixture(scope="module")
def train_kwds(model, data_config, use_train_aug,epochs,workers,batch,lr, imgsz, device, cosine_annealing, 
               square_training, resume_training, no_mosaic, weights, seed):
    """Fixture to return arbitrary keyword arguments for predictions."""
    pred_kwds = {
        'model': model,
        'data_config': data_config,
        'use_train_aug': use_train_aug,
        'device':device,
        'epochs': epochs,
        'workers': workers,
        'batch': batch,
        'lr': lr,
        'imgsz':imgsz,
        'no_mosaic': no_mosaic,
        'cosine_annealing': cosine_annealing,
        'weights': weights,
        'resume_training': resume_training,
        'square_training': square_training,
        'seed': seed }
    return {k: v for k, v in pred_kwds.items() if v is not None}

@pytest.fixture(scope="module")
def trained_model_path(train_kwds):
    result = api.train(**train_kwds)
    saved_model_path = result.split(' ')[-1]
    yield saved_model_path
    
#####################################################
#  Fixtures to test predict function
#####################################################

@pytest.fixture(scope="module", params=fnmatch.filter(DATA_FILES, "*.jpg") + fnmatch.filter(DATA_FILES, "*.png"))
def input(request):
    """Fixture to return input_file argument to test."""
    file = str(Path(configs.DATA_PATH) / request.param)
    content_type = 'application/octet-stream'
    return UploadedFile('input', file, content_type, request.param)


@pytest.fixture(scope="module", params=MODELS_CKPT)
def model_ckpt(request):
    """Fixture to return model checkpoint argument to test."""
    return request.param

@pytest.fixture(scope="module", params=['image/png', 'application/json'])
def accept(request):
    """Fixture to return accept arguments to test."""
    return request.param


@pytest.fixture(scope="module", params=[0.4, 0.5, 0.8])
def threshold(request):
    """Fixture to return threshold argument to test."""
    return request.param


@pytest.fixture(scope="module", params=[640, 320])
def imgsz(request):
    """Fixture to return image size argument to test."""
    return request.param


@pytest.fixture(scope="module", params=[False, True])
def device(request):
    """Fixture to return gpu flag argument to test."""
    return request.param

@pytest.fixture(scope="module", params=[False, True])
def no_labels(request):
    """Fixture to return  no label argument to test."""
    return request.param

@pytest.fixture(scope="module", params=[False, True])
def square_img(request):
    """Fixture to return square image argument to test."""
    return request.param

@pytest.fixture(scope="module", params=['2023-05-10_121810'])
def timestamp(request):
    """Fixture to return square image argument to test."""
    return request.param

@pytest.fixture(scope="module", params='rshare')
def remote_name(request):
    return request.params

@pytest.fixture(scope="module", params=[configs.REMOT_PATH, os.path.join(configs.REMOT_PATH, '2023-05-10_121810')])
def remote_directory(request):
    return  request.params

@pytest.fixture(scope="module", params='models/test')
def local_directory(request):
    os.makedirs(request.params, exist_ok=True)
    return  request.params

@pytest.fixture(scope="module")
def pred_kwds(input, timestamp, threshold, imgsz, device, no_labels, square_img, accept):
    """Fixture to return arbitrary keyword arguments for predictions."""
    pred_kwds = {
        'input': input,
        'timestamp': timestamp,
        'threshold':threshold,
        'imgsz':imgsz,
        'device':device,
        'no_labels':no_labels,
        'square_img':square_img,
        'accept':accept
    }
    return {k: v for k, v in pred_kwds.items() if v is not None}

@pytest.fixture(scope="module")
def test_predict(pred_kwds):
    """Test the predict function."""
    result = api.predict(**pred_kwds)
    return result, pred_kwds['accept']
        



     
        