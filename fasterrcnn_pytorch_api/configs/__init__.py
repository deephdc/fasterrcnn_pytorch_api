"""Configuration loader for rnacontactmap."""
import configparser
import os
import pathlib
from importlib.metadata import metadata as _metadata

#from webdav4.client import Client
homedir = os.path.dirname(os.path.normpath(os.path.dirname(__file__)))


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
    BASE_DIR = os.getenv("BASE_DIR", default=settings['base_dir']['path'])
    # Selbstaufsicht requires currently the setup of DATA_PATH env variable
    if os.path.isabs(BASE_DIR):
         os.environ["BASE_DIR"] = BASE_DIR
    else:
        os.environ["BASE_DIR"] = os.path.abspath(os.path.join(homedir, BASE_DIR))
        
except KeyError as err:
    raise RuntimeError("Undefined configuration for base_dir") from err

try:  # Configure input files for testing and possible training
    DATA_PATH = os.getenv("DATA_PATH", default=settings['data']['path'])
    # Selbstaufsicht requires currently the setup of DATA_PATH env variable
    DATA_PATH =os.path.join(BASE_DIR, DATA_PATH)
    os.environ["DATA_PATH"] =DATA_PATH
except KeyError as err:
    raise RuntimeError("Undefined configuration for [data]path") from err

try:  # Local path for caching   sub/models
    MODEL_DIR = os.getenv("MODEL_DIR", settings['model_dir']['path'])
    MODEL_DIR = os.path.join(BASE_DIR, MODEL_DIR)
    os.environ["MODEL_DIR"] = MODEL_DIR
except KeyError as err:
    raise RuntimeError("Undefined configuration for model path") from err
 

try:  # Local path for caching   sub/models
    REMOT_URL = os.getenv("REMOT", settings['remote']['url'])
    os.environ["REMOT_URL"] = REMOT_URL 
except KeyError as err:
    raise RuntimeError("Undefined configuration for url") from err


