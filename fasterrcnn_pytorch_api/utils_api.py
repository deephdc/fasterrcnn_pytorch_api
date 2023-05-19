"""
This file gathers some functions that have proven to be useful
across projects. They are not strictly need for integration
but users might want nevertheless to take advantage from them.
"""

from functools import wraps
import os
import zipfile

from aiohttp.web import HTTPBadRequest
import requests
from fasterrcnn_pytorch_api import configs

 

def copy_checkpoint_from_url(public_url,  local_folder_path):
  
    response = requests.get(public_url)

# Check if the request was successful
    if response.status_code == 200:
 
        if not os.path.exists(local_folder_path):
            os.makedirs(local_folder_path)


        file_links = set()
        for line in response.text.splitlines():            
             if "download" in line :
                  start_index = line.index("https://")
                  end_index = line.index("\"", start_index)
                  file_links.add(line[start_index:end_index])
        

       
        for file_link in file_links:
             file_name = os.path.basename(file_link)
             file_path = os.path.join(local_folder_path, file_name+'.zip')
             file_response = requests.get(file_link)

        
             if file_response.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(file_response.content)
                    print(f"File '{file_name}' downloaded successfully!")
             else:
        
                print("Error downloading file '{file_name}': {file_response.status_code}")           

    else:
         raise RuntimeError("Error downloading folder: {}".format(response.status_code))

    zip_file_path = os.path.join(local_folder_path, f"{file_name}.zip")
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(local_folder_path)

    os.remove(zip_file_path)


def download_model_from_url(  timestamp  ):
    """
    Downloads the final model from nextcloud to a specified checkpoint path.

   Args:
    timestamp_type (str): Type of timestamp (downstrram or XGB).
    task (str): The name of the head task for downstearm model.
    timestamp (str): The timestamp of the model on Nextcloud.
    final_model (str): The name of the checkpint file.
    ckpt_path (str): The path to the checkpoint directory where the model will be put after downloading.

    Returns:
    None

    Raises:
    Exception: If no files were copied to the checkpoint directory after downloading the model from the URL.

    """
    local_path=configs.MODEL_DIR
    print(local_path)
    
    ckpt_path=os.path.join(local_path, timestamp)
    print(ckpt_path)

    if timestamp not in os.listdir(local_path) :
        print('downloading the model')
    
        url=configs.REMOT_URL
        print(url)
        copy_checkpoint_from_url(url, local_path)

        if 'best_model.pth' not in os.listdir(ckpt_path):
            raise Exception(f"No files were copied to {ckpt_path}")

        print(f"The model for {timestamp} was copied to {ckpt_path}")
    else:
        print(f"Skipping download for {timestamp} as the model already exists in {ckpt_path}")
        
if __name__=='__main__':

        download_model_from_url('2023-05-10_121810')
   

 
