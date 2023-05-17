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

from fasterrcnn_pytorch_api import configs, fields, utils
from fasterrcnn_pytorch_api.scripts import inference
from fasterrcnn_pytorch_api.scripts.train import main

logger = logging.getLogger('__name__')

def get_metadata():
    """Returns a dictionary containing metadata information about the module.

    Returns:
        A dictionary containing metadata information required by DEEPaaS.
    """
    metadata = {
        'authors': configs.MODEL_METADATA.get("author"),
        'description': configs.MODEL_METADATA.get("summary"),
        'license': configs.MODEL_METADATA.get("license"),
        'version': configs.MODEL_METADATA.get("version") 
        }
    logger.debug("Package model metadata: %d", metadata)
    return  metadata

def get_train_args():
    """Return the arguments that are needed to perform a  training.

    Returns:
        Dictionary of webargs fields.
      """
    predict_args=fields.TrainArgsSchema().fields
    logger.debug("Web arguments: %d", predict_args) 
    return  predict_args

def get_predict_args():
    """Return the arguments that are needed to perform a  prediciton.

    Returns:
        Dictionary of webargs fields.
      """
    predict_args=fields.PredictArgsSchema().fields
    logger.debug("Web arguments: %d", predict_args)
    return predict_args

def  train(**args):
    """Performs training on the dataset.
    Args:
        **args:   keyword arguments from get_train_args.
    Returns:
        path to the trained model
    """
 
    timestamp=datetime.now().strftime('%Y-%m-%d_%H%M%S')
    os.mkdir(os.path.join(configs.MODEL_DIR,timestamp))
    args['name']=os.path.join(configs.MODEL_DIR,timestamp)
    args['data_config']=os.path.join(configs.DATA_PATH, args['data_config'])
    main(args)
    return {f'model was saved in {args["name"]}'}


def predict(**args):
    """Performs inference  on an input image.
    Args:
        **args:   keyword arguments from get_predict_args.
    Returns:
        either a json file or png image with bounding box 
    """
    utils.download_model_from_url( args['timestamp'])
    args['weights']=os.path.join(configs.MODEL_DIR,args['timestamp'], 'best_model.pth')
    with tempfile.TemporaryDirectory() as tmpdir: 
        for f in [args['input']]:
           shutil.move(f.filename, tmpdir + F'/{f.original_filename}')
        args['input'] =[os.path.join(tmpdir,t) for t in os.listdir(tmpdir)]
        outputs, buffer=inference.main(args)
        
        if args['accept']== 'image/png':
             return buffer
        else:
            return   outputs

if __name__=='__main__':
     args={'model': 'fasterrcnn_convnext_small',
           'data_config':'brackish.yaml',
           'use_train_aug':False,
           'device':True,
           'epochs':20,
           'workers':4,
           'batch':1,
           'lr':0.001,
           'imgsz':640,
           'no_mosaic':True,
           'use_train_aug':False,
           'cosine_annealing':True,
           'weights':None,
           'resume_training':False,
           'square_training':False,
           'seed':0
           }
     train(**args)
# def warm():
#     pass
