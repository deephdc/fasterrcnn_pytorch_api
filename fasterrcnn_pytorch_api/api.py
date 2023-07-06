# -*- coding: utf-8 -*-
"""
Functions to integrate your model with the DEEPaaS API.
It's usually good practice to keep this file minimal, only performing the interfacing
tasks. In this way you don't mix your true code with DEEPaaS code and everything is
more modular. That is, if you need to write the predict() function in api.py, you
would import your true predict function and call it from here (with some processing /
postprocessing in between if needed).
For example:

    import mycustomfile

    def predict(**kwargs):
        args = preprocess(kwargs)
        resp = mycustomfile.predict(args)
        resp = postprocess(resp)
        return resp

To start populating this file, take a look at the docs [1] and at a canonical exemplar
module [2].

[1]: https://docs.deep-hybrid-datacloud.eu/
[2]: https://github.com/deephdc/demo_app
"""

import logging 
import os
import shutil
import tempfile
from datetime import datetime 

from fasterrcnn_pytorch_api import configs, fields, utils_api
from fasterrcnn_pytorch_api.scripts import inference
from fasterrcnn_pytorch_api.scripts.train import main

logger = logging.getLogger('__name__')

def get_metadata():
    """
    Returns a dictionary containing metadata information about the module.

    Returns:
        A dictionary containing metadata information required by DEEPaaS.
    """
    metadata = {
        'authors': configs.MODEL_METADATA.get("author"),
        'description': configs.MODEL_METADATA.get("summary"),
        'license': configs.MODEL_METADATA.get("license"),
        'version': configs.MODEL_METADATA.get("version"),
        'checkpoints_local': utils_api.ls_local(),
        'checkpoints_remote': utils_api.ls_remote(),
        }
    logger.debug("Package model metadata: %d", metadata)
    return  metadata

def get_train_args():
    """
    Return the arguments that are needed to perform a  training.

    Returns:
        Dictionary of webargs fields.
      """
    train_args=fields.TrainArgsSchema().fields
    logger.debug("Web arguments: %d", train_args) 
    return  train_args

def get_predict_args():
    """
    Return the arguments that are needed to perform a  prediciton.

    Args:
        None

    Returns:
        Dictionary of webargs fields.
    """
    predict_args=fields.PredictArgsSchema().fields
    logger.debug("Web arguments: %d", predict_args)
    return predict_args

def  train(**args):
    """
    Performs training on the dataset.
    Args:
        **args: keyword arguments from get_train_args.
    Returns:
        path to the trained model
    """
    assert not (args.get('resume_training', False) and not args.get('weights')), \
    "weights argument should not be empty when resume_training is True"
    if args['weights'] is not None:
        args['weights']=os.path.join(configs.MODEL_DIR, args['weights'], 'last_model.pth')

    timestamp=datetime.now().strftime('%Y-%m-%d_%H%M%S')
    ckpt_path=os.path.join(configs.MODEL_DIR, timestamp)
    os.makedirs(ckpt_path, exist_ok=True)
    args['name']=ckpt_path
    args['data_config']=os.path.join(configs.DATA_PATH, args['data_config'])
    main(args)
    return {f'model was saved in {args["name"]}'}


def predict(**args):
    """
    Performs inference  on an input image.
    Args:
        **args:   keyword arguments from get_predict_args.
    Returns:
        either a json file or png image with bounding box 
    """
    
    
    if args['timestamp'] is not None:
        utils_api.download_model_from_nextcloud(args['timestamp'])
        args['weights']=os.path.join(configs.MODEL_DIR, args['timestamp'], 'best_model.pth')
        if os.path.exists(args['weights']):
            print("best_model.pth exists at the specified path.")        
    else:        
         args['weights']=None

    with tempfile.TemporaryDirectory() as tmpdir: 
        for f in [args['input']]:
           shutil.copy(f.filename, tmpdir + F'/{f.original_filename}')
        args['input'] =[os.path.join(tmpdir,t) for t in os.listdir(tmpdir)]
        outputs, buffer=inference.main(args)
        
        if args['accept']== 'image/png':
             return buffer
        else:
            return   outputs

if __name__=='__main__':
     args={'model': 'fasterrcnn_convnext_small',
           'data_config':  'brackish.yaml',
           'use_train_aug': True,
           'aug_training_option':{
                                            'blur': {'p':0.3, 'blur_limit':3},
                                            'motion_blur': {'p':0.1, 'blur_limit':3},
                                            'median_blur': {'p':0.5, 'blur_limit':3},
                                            'to_gray': {'p':0.1},
                                            'random_brightness_contrast': {'p':0.1},
                                            'color_jitter': {'p':0.2},
                                            'random_gamma': {'p':0.1}, 
                                            'horizontal_flip': {'p':1},
                                            'vertical_flip': {'p':1},
                                            'rotate': {'limit':60},
                                            'shift_scale_rotate': {'shift_limit':0.1, 'scale_limit':0.1, 'rotate_limit':30},
                                            'Cutout': {'num_holes':5, 'max_h_size':3, 'max_w_size':5, 'fill_value':0, 'p':1},
                                            'ChannelShuffle': {'p':1},
                                        },
           'device': True,
           'epochs': 3,
           'workers': 4,
           'batch': 1,
           'lr': 0.001,
           'imgsz': 640,

           'no_mosaic': True,
           'cosine_annealing': False,
           'weights': '2023-05-10_121810',
           'resume_training': True,
           'square_training': False,
           'seed':0,
           'eval_n_epochs':3
           }
     train(**args)
     input='/home/se1131/EGI/fasterrcnn_pytorch_api/data/submarine_det/test.jpg'
     from deepaas.model.v2.wrapper import UploadedFile
     pred_kwds = {
        'input': UploadedFile('input', input,  'application/octet-stream','input' ),
        'timestamp':None,
        'model':  'fasterrcnn_convnext_tiny',
        'threshold':0.5,
        'imgsz':640,
        'device':False,
        'no_labels':False,
        'square_img':False,
        'accept': 'application/json'
    }

    

