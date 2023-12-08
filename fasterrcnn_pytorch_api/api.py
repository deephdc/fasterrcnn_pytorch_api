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

import argparse
import json
import logging 
import os
import shutil
import tempfile
from datetime import datetime 
from aiohttp.web import HTTPException
from deepaas.model.v2.wrapper import UploadedFile

from fasterrcnn_pytorch_api import configs, fields, utils_api
from fasterrcnn_pytorch_api.scripts import inference
from fasterrcnn_pytorch_api.scripts.mlflow_train import main as mlflow_train_model
from fasterrcnn_pytorch_api.scripts.train import main as train_model

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

def train(**args):
    """
    Performs training on the dataset.

    Args:
        **args: keyword arguments from get_train_args.

    Returns:
        path to the trained model
    """
    try:
        logger.info("Training model...")
        logger.debug("Train with args: %s", args)
        if args.get("resume_training", False) and not args.get(
            "weights"
        ):
            logger.error(
                "weights argument should not be empty when"
                "resume_training is True"
            )
            raise ValueError(
                "Weights argument is missing for resumed training"
            )


        if args["weights"] is not None:
            args["weights"] = os.path.join(
                args["weights"], "last_model.pth"
            )
            args["weights"] = utils_api.validate_and_modify_path(
                args["weights"], configs.MODEL_DIR
            )

        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        ckpt_path = os.path.join(configs.MODEL_DIR, timestamp)
        os.makedirs(ckpt_path, exist_ok=True)
        args["name"] = ckpt_path
        args["data_config"] = utils_api.validate_and_modify_path(
            args["data_config"], configs.DATA_PATH
        )
        if args["disable_mlflow"]: 
          train_model(args)
          return {'result': f'model was saved in {args["name"]}'}
        else:
            run_id= mlflow_train_model(args)
            return {'result': f'model was saved in your mlflow server with run_id: {run_id}'}
    except Exception as err:
        logger.critical(err, exc_info=True)
        raise HTTPException(reason=err) from err



def predict(**args):
    """
    Performs inference  on an input image.
    Args:
        **args:   keyword arguments from get_predict_args.
    Returns:
        either a json file or png image with bounding box 
    """
    utils_api.download_model_from_nextcloud(args['timestamp'])

    args['weights']=os.path.join(configs.MODEL_DIR, args['timestamp'], 'best_model.pth')

    with tempfile.TemporaryDirectory() as tmpdir: 
        for f in [args['input']]:
           shutil.copy(f.filename, tmpdir + F'/{f.original_filename}')
        args['input'] =[os.path.join(tmpdir,t) for t in os.listdir(tmpdir)]
        outputs, buffer=inference.main(args)
        
        if args['accept']== 'image/png':
             return buffer
        else:
            return   outputs

def main():
    """
    Runs above-described methods from CLI
    uses: python3 path/to/api.py method --arg1 ARG1_VALUE
     --arg2 ARG2_VALUE
    """
    method_dispatch = {
        "get_metadata": get_metadata,
        "predict": predict,
        "train": train,
    }

    chosen_method = args.method
    logger.debug("Calling method: %s", chosen_method)
    if chosen_method in method_dispatch:
        method_function = method_dispatch[chosen_method]

        if chosen_method == "get_metadata":
            results = method_function()
        else:
            logger.debug("Calling method with args: %s", args)
            del vars(args)["method"]
            if hasattr(args, "input"):
                file_extension = os.path.splitext(args.input)[1]
                args.input = UploadedFile(
                    "input",
                    args.input,
                    "application/octet-stream",
                    f"input{file_extension}",
                )
            results = method_function(**vars(args))

        print(json.dumps(results))
        logger.debug("Results: %s", results)
        return results
    else:
        print("Invalid method specified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model parameters", add_help=False
    )
    cmd_parser = argparse.ArgumentParser()
    subparsers = cmd_parser.add_subparsers(
        help='methods. Use "api.py method --help" to get more info',
        dest="method",
    )
    get_metadata_parser = subparsers.add_parser(
        "get_metadata", help="get_metadata method", parents=[parser]
    )

    predict_parser = subparsers.add_parser(
        "predict", help="commands for prediction", parents=[parser]
    )

    utils_api.add_arguments_from_schema(
        fields.PredictArgsSchema(), predict_parser
    )

    train_parser = subparsers.add_parser(
        "train", help="commands for training", parents=[parser]
    )
    utils_api.add_arguments_from_schema(
        fields.TrainArgsSchema(), train_parser
    )

    args = cmd_parser.parse_args()

    main()
