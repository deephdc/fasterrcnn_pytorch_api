# -*- coding: utf-8 -*-
"""
Functions to integrate your model with the DEEPaaS API.
It's usually good practice to keep this file minimal, only
performing the interfacing tasks. In this way, you don't mix
your true code with DEEPaaS code and everything is more modular.
That is, if you need to write the predict() function in api.py, you
would import your true predict function and call it from here
(with some processing /postprocessing in between if needed).
For example:

    import mycustomfile

    def predict(**kwargs):
        args = preprocess(kwargs)
        resp = mycustomfile.predict(args)
        resp = postprocess(resp)
        return resp

To start populating this file, take a look at the docs [1] and
at a canonical exemplar module [2].

[1]: https://docs.deep-hybrid-datacloud.eu/
[2]: https://github.com/deephdc/demo_app
"""

import logging
import os
import shutil
import tempfile
from datetime import datetime
from multiprocessing import Process
from aiohttp.web import HTTPException
import ast
import argparse
import json

import wandb
from fasterrcnn_pytorch_api import configs, fields, utils_api
from fasterrcnn_pytorch_api.scripts.train import main
from fasterrcnn_pytorch_api.scripts import combineinfer

logger = logging.getLogger(__name__)


def get_metadata():
    """
    Returns a dictionary containing metadata information about the module.

    Returns:
        A dictionary containing metadata information required by DEEPaaS.
    """
    metadata = {
        "authors": configs.MODEL_METADATA.get("author"),
        "description": configs.MODEL_METADATA.get("summary"),
        "license": configs.MODEL_METADATA.get("license"),
        "version": configs.MODEL_METADATA.get("version"),
        "checkpoints_local": utils_api.ls_local(),
        "checkpoints_remote": utils_api.ls_remote(),
    }
    logger.debug("Package model metadata: %s", metadata)
    return metadata


def get_train_args():
    """
    Return the arguments that are needed to perform a training.

    Returns:
        Dictionary of webargs fields.
    """
    train_args = fields.TrainArgsSchema().fields
    logger.debug("Web arguments: %s", train_args)
    return train_args


def get_predict_args():
    """
    Return the arguments that are needed to perform a prediction.

    Args:
        None

    Returns:
        Dictionary of webargs fields.
    """
    predict_args = fields.PredictArgsSchema().fields
    logger.debug("Web arguments: %s", predict_args)
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

        if not args["disable_wandb"]:
            wandb.login(key=configs.WANDB_TOKEN)

        if args["weights"] is not None:
            args["weights"] = os.path.join(
                configs.MODEL_DIR, args["weights"], "last_model.pth"
            )

        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        ckpt_path = os.path.join(configs.MODEL_DIR, timestamp)
        os.makedirs(ckpt_path, exist_ok=True)
        args["name"] = ckpt_path
        args["data_config"] = os.path.join(
            configs.DATA_PATH, args["data_config"]
        )

        p = Process(
            target=utils_api.launch_tensorboard,
            args=(configs.MONITOR_PORT, configs.MODEL_DIR),
            daemon=True,
        )
        p.start()
        main(args)
        return {f'model was saved in {args["name"]}'}
    except Exception as err:
        raise HTTPException(reason=err) from err


def predict(**args):
    """
    Performs inference on an input image.

    Args:
        **args: keyword arguments from get_predict_args.

    Returns:
        either a JSON file, PNG image or video with bounding boxes.
    """
    try:
        logger.debug("Predict with args: %s", args)
        timestamp = args.get("timestamp")
        if timestamp is not None:
            if ast.literal_eval(configs.USE_RCLONE):
                logger.error(
                    "Set the rclone configuration in settings.ini"
                )
                utils_api.download_model_from_nextcloud(timestamp)
            if timestamp not in os.listdir(
                configs.MODEL_DIR
            ):
                raise ValueError(
                    f"Timestamp '{timestamp}' not found in '{configs.MODEL_DIR}'"
                )
            args["weights"] = os.path.join(
                configs.MODEL_DIR, timestamp, "best_model.pth"
            )

        else:
            args["weights"] = None

        with tempfile.TemporaryDirectory() as tmpdir:
            args["input"] = [args["input"]]
            file_format = utils_api.check_input_type(
                args["input"][0].original_filename
            )
            for f in args["input"]:
                shutil.copy(
                    f.filename, tmpdir + "/" + f.original_filename
                )
            args["input"] = [
                os.path.join(tmpdir, t) for t in os.listdir(tmpdir)
            ]
            engine = combineinfer.InferenceEngine(args)
            json_string, buffer = engine.infer(file_format, **args)
            logger.debug("Response json_string: %d", json_string)
            logger.debug("Response buffer: %d", buffer)

            if args["accept"] == "application/json":
                return json_string
            else:
                return buffer

    except Exception as err:
        raise HTTPException(reason=err) from err
def main():
    """
    Runs above-described methods from CLI
    """
    method_dispatch = {
        'get_metadata': get_metadata,
        'predict': predict,
        'train': train
    }

    chosen_method = args.method
    logger.debug("Calling method: %s", chosen_method)
    if chosen_method in method_dispatch:
        method_function = method_dispatch[chosen_method]

        if chosen_method == 'get_metadata':
            results = method_function()
        else:
            logger.debug("Calling method with args: %s", args)
            results = method_function(**vars(args))
            

        print(json.dumps(results))
        logger.debug("Results: %s", results)
        return results
    else:
        print("Invalid method specified.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model parameters', 
                                     add_help=False)
    cmd_parser = argparse.ArgumentParser()
    subparsers = cmd_parser.add_subparsers(
                            help='methods. Use \"api.py method --help\" to get more info', 
                            dest='method')          
    get_metadata_parser = subparsers.add_parser('get_metadata', 
                                         help='get_metadata method',
                                         parents=[parser])                                               
    
    predict_parser = subparsers.add_parser('predict', 
                                           help='commands for prediction',
                                           parents=[parser]) 

    utils_api.add_arguments_from_schema(fields.PredictArgsSchema(), predict_parser) 

    train_parser = subparsers.add_parser('train', 
                                         help='commands for training',
                                         parents=[parser])    
    utils_api.add_arguments_from_schema(fields.TrainArgsSchema(), train_parser)                                                                        

    
  
    args = cmd_parser.parse_args()
    
    main()