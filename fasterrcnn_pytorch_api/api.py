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

import os
from pathlib import Path
import shutil
import tempfile
import pkg_resources
from datetime import datetime 

from fasterrcnn_pytorch_api.misc import _catch_error
import fasterrcnn_pytorch_api.config as cfg
from fasterrcnn_pytorch_api.scripts import inference
from fasterrcnn_pytorch_api.scripts.train import main


BASE_DIR = Path(__file__).resolve().parents[1]


@_catch_error
def get_metadata():
    """
    DO NOT REMOVE - All modules should have a get_metadata() function
    with appropriate keys.
    """
    distros = list(pkg_resources.find_distributions(str(BASE_DIR), only=True))
    if len(distros) == 0:
        raise Exception("No package found.")
    pkg = distros[0]  # if several select first

    meta_fields = {
        "name": None,
        "version": None,
        "summary": None,
        "home-page": None,
        "author": None,
        "author-email": None,
        "license": None,
    }
    meta = {}
    for line in pkg.get_metadata_lines("PKG-INFO"):
        line_low = line.lower()  # to avoid inconsistency due to letter cases
        for k in meta_fields:
            if line_low.startswith(k + ":"):
                _, value = line.split(": ", 1)
                meta[k] = value

    return meta


def get_train_args():
     return  cfg.training_args


def train(**args):
        timestamp=datetime.now().strftime('%Y-%m-%d_%H%M%S')
        os.mkdir(os.path.join(cfg.MODEL_DIR,timestamp))
        args['name']=os.path.join(cfg.MODEL_DIR,timestamp)
        args['data_config']=os.path.join(cfg.DATASET_DIR, args['data_config'])
        main(args)
        return {f'model was save'}

def get_predict_args():
     return cfg.predict_args

@_catch_error
def predict(**args):
    args['input'] = [args['input']]
    with tempfile.TemporaryDirectory() as tmpdir: 
        for f in args['input_files']:
           shutil.move(f.filename, tmpdir + F'/{f.original_filename}')
           args['input'] =[ os.path.join(tmpdir,t) for t in os.listdir(tmpdir)]

    outputs, buffer=inference.main(args)
    if args['accept']== 'image/png':
             return buffer
    else:
            return   outputs
if __name__=='__main__':
     args={'model': 'fasterrcnn_convnext_small',
           'data_config':'/home/se1131/fasterrcnn_pytorch_api/data/submarine_det/brackish.yaml',
           'use_train_aug':False,
           'device':True,
           'epochs':1,
           'workers':4,
           'batch':2,
           'lr':0.001,
           'imgsz':640,
           'vis_transformed':False,
           'no_mosaic':True,
           'use_train_aug':False,
           'SAVE_VALID_PREDICTION_IMAGES':False,
           'cosine_annealing':True,
           'weights':None,
           'resume_training':False,
           'square_training':False,
           'seed':0

           }
     train(**args)
# def warm():
#     pass
#
#

#
#

#
#
 
#
# 
