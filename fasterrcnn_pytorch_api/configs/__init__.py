"""Configuration loader for FasterRCNN."""
import ast
import configparser
import os
import pathlib
from importlib.metadata import metadata as _metadata

homedir = os.path.dirname(os.path.normpath(os.path.dirname(__file__)))
base_dir = os.path.dirname(os.path.abspath(homedir))

# Get configuration from user env and merge with pkg settings
SETTINGS_FILE = pathlib.Path(__file__).parent / "settings.ini"
SETTINGS_FILE = os.getenv(
    "fasterrcnn-pytorch-training-pipeline_SERRING",
    default=SETTINGS_FILE,
)
settings = configparser.ConfigParser()
settings.read(SETTINGS_FILE)


def resolve_path(base_dir):
    if os.path.isabs(base_dir):
        return base_dir
    else:
        return os.path.abspath(os.path.join(homedir, base_dir))


try:  # Configure model metadata from pkg metadata
    MODEL_NAME = os.getenv(
        "MODEL_NAME", default=settings["model"]["name"]
    )
    MODEL_METADATA = _metadata(MODEL_NAME)
except KeyError as err:
    raise RuntimeError(
        "Undefined configuration for [model] name"
    ) from err


try:  # Configure input files for testing and possible training
    DATA_PATH_DEFAULT = os.path.join(
        base_dir, settings["data"]["path"]
    )
    DATA_PATH = os.getenv("DATA_PATH", default=DATA_PATH_DEFAULT)
except KeyError as err:
    raise RuntimeError(
        "Undefined configuration for [data]path"
    ) from err

try:
    MODEL_DIR_DEFAULT = os.path.join(
        base_dir, settings["model_dir"]["path"]
    )
    MODEL_DIR = os.getenv("MODEL_DIR", default=MODEL_DIR_DEFAULT)
except KeyError as err:
    raise RuntimeError(
        "Undefined configuration for model path"
    ) from err

try:  # remote path to test model
    REMOTE_PATH = os.getenv("REMOTE", settings["remote"]["path"])
    os.environ["REMOTE_PATH"] = REMOTE_PATH
except KeyError as err:
    raise RuntimeError(
        "Undefined configuration for Remotepath"
    ) from err

try:  # Port for monitoring using tensorboard
    MONITOR_PORT = os.getenv(
        "MONITOR_PORT", settings["monitorPORT"]["port"]
    )
    os.environ["MONITOR_PORT"] = MONITOR_PORT
except KeyError as err:
    raise RuntimeError(
        "Undefined Monitor port for tensorboar"
    ) from err

try:
    WANDB_TOKEN = os.getenv(
        "wandb_token", settings["wandb_token"]["token"]
    )
except KeyError as err:
    raise RuntimeError(
        "Undefined configuration for WANDB_TOKEN"
    ) from err


try:
    BACKBONES = os.getenv("REMOT", settings["backbones"]["names"])
    if isinstance(BACKBONES, str):
        # Parse the string as a list of strings
        BACKBONES = ast.literal_eval(BACKBONES)
except KeyError as err:
    raise RuntimeError(
        "Undefined configuration for backbones"
    ) from err

try:
    DATA_AUG_OPTION = os.getenv(
        "data_augmentation_options",
        settings["data_augmentation_options"]["names"],
    )
    if isinstance(DATA_AUG_OPTION, str):
        # Parse the string as a list of strings
        DATA_AUG_OPTION = ast.literal_eval(DATA_AUG_OPTION)
except KeyError as err:
    raise RuntimeError(
        "Undefined configuration for data augmentation options"
    ) from err

try:
    DATA_AUG_OPTION = os.getenv(
        "data_augmentation_options",
        settings["data_augmentation_options"]["names"],
    )
    if isinstance(DATA_AUG_OPTION, str):
        # Parse the string as a list of strings
        DATA_AUG_OPTION = ast.literal_eval(DATA_AUG_OPTION)
except KeyError as err:
    raise RuntimeError(
        "Undefined configuration for data augmentation options"
    ) from err

try:
    USE_RCLONE = os.getenv(
        "use_rclone", settings["use_rclone"]["value"]
    )
except KeyError as err:
    raise RuntimeError(
        "Undefined configuration for use_rclone options"
    ) from err

