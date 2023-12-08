"""
This file gathers some functions that have proven to be useful
across projects. They are not strictly need for integration
but users might want nevertheless to take advantage from them.
"""

import logging
import os
import subprocess  # nosec B404
import argparse
from marshmallow import fields

from fasterrcnn_pytorch_api import configs

logger = logging.getLogger("__name__")


def ls_local():
    """
    Utility to return a list of current models stored in the local folders
    configured for cache.

    Returns:
        A list of strings.
    """
    logger.debug("Scanning at: %s", configs.MODEL_DIR)
    dirscan = os.scandir(configs.MODEL_DIR)
    return [entry.name for entry in dirscan if entry.is_dir()]


def ls_remote():
    """
    Utility to return a list of current backbone models stored in the
    remote folder configured in the backbone url.

    Returns:
        A list of strings.
    """
    remote_directory = configs.REMOTE_PATH
    return list_directories_with_rclone("rshare", remote_directory)


def list_directories_with_rclone(remote_name, directory_path):
    """
    Function to list directories within a given directory in Nextcloud
    using rclone.

    Args:
        remote_name (str): Name of the configured Nextcloud remote in rclone.
        directory_path (str): Path of the parent directory to list the
            directories from.

    Returns:
        list: List of directory names within the specified parent directory.
    """
    command = ["rclone", "lsf", remote_name + ":" + directory_path]
    result = subprocess.run(
        command, capture_output=True, text=True, shell=False
    )  # nosec B603

    if result.returncode == 0:
        directory_names = result.stdout.splitlines()
        directory_names = [
            d.rstrip("/") for d in directory_names if d[0].isdigit()
        ]
        return directory_names
    else:
        print("Error executing rclone command:", result.stderr)
        return []


def download_model_from_nextcloud(timestamp):
    """
    Downloads the final model from nextcloud to a specified checkpoint path.

    Args:
        timestamp (str): The timestamp of the model on Nextcloud.

    Returns:
       None

    Raises:
       Exception: If no files were copied to the checkpoint directory after
       downloading the model from the URL.

    """
    logger.debug("Scanning at: %s", timestamp)
    logger.debug("Scanning at: %s", configs.REMOTE_PATH)
    local_path = configs.MODEL_DIR
    ckpt_path = os.path.join(local_path, timestamp)

    if timestamp not in os.listdir(local_path):
        print("downloading the chekpoint from nextcloud")
        remote_directory = configs.REMOTE_PATH
        model_path = os.path.join(remote_directory, timestamp)
        download_directory_with_rclone(
            "rshare", model_path, local_path
        )

        if "best_model.pth" not in os.listdir(ckpt_path):
            raise Exception(f"No files were copied to {ckpt_path}")

        print(f"The model for {timestamp} was copied to {ckpt_path}")

    else:
        print(
            f"Skipping download for {timestamp} as the model already exists in"
            f" {ckpt_path}"
        )


def download_directory_with_rclone(
    remote_name, remote_directory, local_directory
):
    """
    Function to download a directory using rclone.

    Args:
        remote_name (str): Name of the configured remote in rclone.
        remote_directory (str): Path of the remote directory to be downloaded.
        local_directory (str): Path of the local directory to save the
        downloaded files.

    Returns:
        None
    """

    command = [
        "rclone",
        "copy",
        remote_name + ":" + remote_directory,
        local_directory,
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, shell=False
    )  # nosec B603

    if result.returncode == 0:
        print("Directory downloaded successfully.")
    else:
        print("Error executing rclone command:", result.stderr)


def launch_tensorboard(port, logdir):
    """
    Function to launch tensorboard.

    Args:
        port (int): Port to use for tensorboard.
        logdir (str): Path to the log directory.

    Returns:
        None
    """
    # host has to be 0.0.0.0 (!),
    # otherwise requests from outside are not accepted
    subprocess.call(
        [
            "tensorboard",
            "--logdir",
            "{}".format(logdir),
            "--port",
            "{}".format(port),
            "--host",
            "127.0.0.0",
        ]
    )  # nosec B603, B607


def check_input_type(file_path):
    # Get the file extension
    file_extension = file_path.split(".")[-1].lower()

    # List of video file extensions
    video_extensions = ["mp4", "avi", "mov", "mkv", "flv", "wmv"]

    # List of image file extensions
    image_extensions = ["jpg", "jpeg", "png", "gif", "bmp", "tiff"]

    if file_extension in video_extensions:
        return "video"
    elif file_extension in image_extensions:
        return "image"
    else:
        return "unknown"


def add_arguments_from_schema(schema, parser):
    """
    Iterates through the fields defined in a schema and adds
    corresponding commandline arguments to the provided
    ArgumentParser object.

    Args:
        schema (marshmallow.Schema): The schema object containing field
        definitions.
        parser (argparse.ArgumentParser): The ArgumentParser object
        to which arguments will be added.

    Returns:
        None
    """
    for field_name, field_obj in schema.fields.items():
        arg_name = f"--{field_name}"

        arg_kwargs = {
            "help": field_name,
        }

        if isinstance(field_obj, fields.Int):
            arg_kwargs["type"] = int
        elif isinstance(field_obj, fields.Bool):
            arg_kwargs["action"] = (
                "store_true"
                if not field_obj.load_default
                else "store_false"
            )

        elif isinstance(field_obj, fields.Float):
            arg_kwargs["type"] = float
        else:
            arg_kwargs["type"] = str

        if field_obj.required:
            arg_kwargs["required"] = True

        if field_obj.load_default and not isinstance(
            field_obj, fields.Bool
        ):
            arg_kwargs["default"] = field_obj.load_default

        if field_obj.metadata.get("description"):
            arg_kwargs["help"] = field_obj.metadata["description"]
        # Debug print statements
        parser.add_argument(arg_name, **arg_kwargs)


def validate_and_modify_path(path, base_path):
    """
    Validate and modify a file path, ensuring it exists

    Args:
        path (str): The input file path to validate.
        base_path (str): The base path to join with 'path' if it
        doesn't exist as-is.

    Returns:
        str: The validated and possibly modified file path.
    """
    if not os.path.isfile(path):
        modified_path = os.path.join(base_path, path)
        if not os.path.isfile(modified_path):
            raise ValueError(
                f"The path {path} does not exist."
                "Please provide a valid path."
            )
        return modified_path
    return path


if __name__ == "__main__":
    from fasterrcnn_pytorch_api import fields as fl

    predict_parser = argparse.ArgumentParser()
    add_arguments_from_schema(fl.TrainArgsSchema(), predict_parser)
    parsed_args = predict_parser.parse_args()
    print(parsed_args)
