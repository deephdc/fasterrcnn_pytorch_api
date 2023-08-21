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
        assert not (
            args.get("resume_training", False)
            and not args.get("weights")
        ), (
            "weights argument should not be empty when resume_training is"
            " True"
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

        if args["timestamp"] is not None:
            utils_api.download_model_from_nextcloud(args["timestamp"])
            args["weights"] = os.path.join(
                configs.MODEL_DIR, args["timestamp"], "best_model.pth"
            )
            if os.path.exists(args["weights"]):
                print("best_model.pth exists at the specified path.")
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


if __name__ == "__main__":
    args = {
        "model": "fasterrcnn_convnext_small",
        "data_config": "submarine_det/brackish.yaml",
        "use_train_aug": True,
        "aug_training_option": configs.DATA_AUG_OPTION,
        "device": True,
        "epochs": 3,
        "workers": 4,
        "batch": 1,
        "lr": 0.001,
        "imgsz": 640,
        "no_mosaic": True,
        "cosine_annealing": False,
        "weights": None,
        "resume_training": False,
        "square_training": False,
        "seed": 0,
        "eval_n_epochs": 3,
        "disable_wandb": True,
    }
    # train(**args)
    input_file = "/srv/yolov8_api/data/mixkit-white-cat-lying" + \
    "-among-the-grasses-seen-up-close-22732-large.mp4"

    from deepaas.model.v2.wrapper import UploadedFile
    pred_kwds = {
        "input": UploadedFile(
            "input",
            input_file,
            "application/octet-stream",
            "input.mp4",
        ),
        "timestamp": "2023-05-10_121810",
        "model": "",
        "threshold": 0.5,
        "imgsz": 640,
        "device": False,
        "no_labels": False,
        "square_img": False,
        "accept": "application/json",
    }
    predict(**pred_kwds)
