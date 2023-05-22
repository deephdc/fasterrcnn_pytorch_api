"""Configuration loader for rnacontactmap."""
import configparser
import os
import pathlib
from importlib.metadata import metadata as _metadata

#from webdav4.client import Client
homedir = os.path.dirname(os.path.normpath(os.path.dirname(__file__)))
base_dir=os.path.dirname(os.path.abspath(homedir))


# Get configuration from user env and merge with pkg settings
SETTINGS_FILE = pathlib.Path(__file__).parent / "settings.ini"
SETTINGS_FILE = os.getenv("fasterrcnn-pytorch-training-pipeline_SERRING", default=SETTINGS_FILE)
settings = configparser.ConfigParser()
settings.read(SETTINGS_FILE)

def resolve_path(base_dir):
    if os.path.isabs(base_dir):
        return base_dir
    else:
        return os.path.abspath(os.path.join(homedir, base_dir))

try:  # Configure model metadata from pkg metadata 
    MODEL_NAME = os.getenv("MODEL_NAME", default=settings['model']['name'])
    MODEL_METADATA = _metadata(MODEL_NAME).json
except KeyError as err:
    raise RuntimeError("Undefined configuration for [model]name") from err


try:  # Configure input files for testing and possible training
    DATA_PATH = os.getenv("DATA_PATH", default=settings['data']['path'])
    # Selbstaufsicht requires currently the setup of DATA_PATH env variable
    DATA_PATH =os.path.join(base_dir, DATA_PATH)
    os.environ["DATA_PATH"] =DATA_PATH
except KeyError as err:
    raise RuntimeError("Undefined configuration for [data]path") from err

try:  # Local path for caching   sub/models
    MODEL_DIR = os.getenv("MODEL_DIR", settings['model_dir']['path'])
    MODEL_DIR = os.path.join(base_dir, MODEL_DIR)
    os.environ["MODEL_DIR"] = MODEL_DIR
except KeyError as err:
    raise RuntimeError("Undefined configuration for model path") from err
 

try:  # remote path sub/models
    REMOT_PATH = os.getenv("REMOT", settings['remote']['path'])
    os.environ["REMOT_PATH"] = REMOT_PATH
except KeyError as err:
    raise RuntimeError("Undefined configuration for Remotepath") from err


