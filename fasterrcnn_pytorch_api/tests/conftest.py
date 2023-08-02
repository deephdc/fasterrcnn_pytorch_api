"""Module for evolution requirements fixtures"""
# pylint: disable=redefined-outer-name

import fnmatch
import json
import os
from pathlib import Path

import pytest
from deepaas.model.v2.wrapper import UploadedFile
import yaml

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

@pytest.fixture(scope="module", params=[os.path.join(configs.DATA_PATH,'submarine_det/brackish.yaml')  ])#FIXME:name of the data config
def data_config(request):
    """Fixture to return data_configs argument to test."""
    return request.param

@pytest.fixture(scope="module")#, params=['2023-05-10_121810')]) 

def weights():
    """Fixture to return  no label argument to test."""
    return None

@pytest.fixture(scope="module", params=['fasterrcnn_convnext_small'])
def model(request):
    """Fixture to return model checkpoint argument to test."""
    return request.param

@pytest.fixture(scope="module", params=[True])
def cosine_annealing(request):
    """Fixture to return cosine annealing argument to test."""
    return request.param

@pytest.fixture(scope="module", params=[False])
def resume_training(request):
    """Fixture to return resume training argument to test."""
    return request.param

@pytest.fixture(scope="module", params=[2])
def eval_n_epochs(request):
    """Fixture to return eval_n_epochs argument to test."""
    return request.param

@pytest.fixture(scope="module", params=[True])
def use_train_aug(request):
    """Fixture to return use train augmentations argument to test."""
    return request.param
   
@pytest.fixture(scope="module", params=[{
    'blur': {'p': 0.1, 'blur_limit': 3},
    'motion_blur': {'p': 0.1, 'blur_limit': 3},
    'median_blur': {'p': 0.1, 'blur_limit': 3},
    'to_gray': {'p': 0.1},
    'random_brightness_contrast': {'p': 0.1},
    'color_jitter': {'p': 0.1},
    'random_gamma': {'p': 0.1}, 
    'horizontal_flip': {'p': 1},
    'vertical_flip': {'p': 1},
    'rotate': {'limit': 45},
    'shift_scale_rotate': {'shift_limit': 0.1, 'scale_limit': 0.1, 'rotate_limit': 30},
    'Cutout': {'num_holes': 0, 'max_h_size': 0, 'max_w_size': '8', 'fill_value': 0, 'p': 0},
    'ChannelShuffle': {'p': 0},
}])
def aug_training_option(request):
    """Fixture to return use train augmentations argument to test."""
    return request.param

@pytest.fixture(scope="module", params=[True])
def square_training(request):
    """Fixture to return square training argument to test."""
    return request.param

@pytest.fixture(scope="module", params=[False])
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

@pytest.fixture(scope="module", params=[4])
def workers(request):
    """Fixture to return number of workers argument to test."""
    return request.param

@pytest.fixture(scope="module", params=[1])
def batch(request):
    """Fixture to return batch size argument to test."""
    return request.param

@pytest.fixture(scope="module", params=[0.001])
def lr(request):
    """Fixture to return learning rate argument to test."""
    return request.param

@pytest.fixture(scope="module", params=[True])
def disable_wandb(request):
    """Fixture to return use train augmentations argument to test."""
    return request.param 
@pytest.fixture(scope="module")
def train_kwds(model, data_config, use_train_aug,aug_training_option,epochs,workers,batch,lr, imgsz, device, cosine_annealing, 
               square_training, resume_training, no_mosaic, weights, seed, eval_n_epochs, disable_wandb):
    """Fixture to return arbitrary keyword arguments for predictions."""
    train_kwds = {
        'model': model,
        'data_config': data_config,
        'use_train_aug': use_train_aug,
        'aug_training_option':aug_training_option,
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
        'seed': seed ,
        'eval_n_epochs':eval_n_epochs,
        'disable_wandb':disable_wandb}
    
    return {k: v for k, v in train_kwds.items()}

@pytest.fixture(scope="module")
def trained_model_path(train_kwds):
    print(train_kwds)
    result = api.train(**train_kwds)
    saved_model_path = str(result).split(' ')[-1].rstrip("'}")
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


@pytest.fixture(scope="module", params=['image/png', 'application/json'])
def accept(request):
    """Fixture to return accept arguments to test."""
    return request.param


@pytest.fixture(scope="module", params=[0.4, 0.5, 0.8])
def threshold(request):
    """Fixture to return threshold argument to test."""
    return request.param


@pytest.fixture(scope="module", params=[640])
def imgsz(request):
    """Fixture to return image size argument to test."""
    return request.param


@pytest.fixture(scope="module", params=[False ])
def device(request):
    """Fixture to return gpu flag argument to test."""
    return request.param

@pytest.fixture(scope="module", params=[False ])
def no_labels(request):
    """Fixture to return  no label argument to test."""
    return request.param

@pytest.fixture(scope="module", params=[False])
def square_img(request):
    """Fixture to return square image argument to test."""
    return request.param

@pytest.fixture(scope="module", params=['2023-05-10_121810', None])
def timestamp(request):
    """Fixture to return square image argument to test."""
    return request.param

@pytest.fixture(scope="module", params='rshare')
def remote_name(request):
    return request.params

 
@pytest.fixture(scope="module")
def pred_kwds(input, timestamp, threshold, model,imgsz, device, no_labels, square_img, accept):
    """Fixture to return arbitrary keyword arguments for predictions."""
    pred_kwds = {
        'input': input,
        'timestamp': timestamp,
        'model': model,
        'threshold':threshold,
        'imgsz':imgsz,
        'device':device,
        'no_labels':no_labels,
        'square_img':square_img,
        'accept':accept
    }
    return {k: v for k, v in pred_kwds.items()}

@pytest.fixture(scope="module")
def test_predict(pred_kwds):
    """Test the predict function."""
    result = api.predict(**pred_kwds)
    return result, pred_kwds['accept']
        



     
        
