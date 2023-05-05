import os
import shutil

# create directories for jpg and xml files
path='/home/se1131/fasterrcnn_pytorch_api/data/submarine_det/test'
jpg_dir=os.path.join(path, 'img')
xml_dir=os.path.join(path, 'labels')
os.makedirs(jpg_dir, exist_ok=True)
os.makedirs(xml_dir, exist_ok=True)

# get list of files in directory
files = os.listdir(path)

# move jpg and xml files to their respective directories
for file in files:
    if file.endswith(".jpg"):
        shutil.move(os.path.join(path,file), jpg_dir)
       
    elif file.endswith(".xml"):
        shutil.move(os.path.join(path,file), xml_dir)
