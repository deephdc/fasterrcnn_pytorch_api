from io import BytesIO
import json
import mlflow
import numpy as np
import cv2
import torch
import glob as glob
import os
import time

from mlflow.tracking.client import MlflowClient
from models.create_fasterrcnn_model import create_model
from utils.annotations import inference_annotations
from utils.transforms import infer_transforms, resize

def collect_all_images(dir_test):
    """
    Function to return a list of image paths.

    :param dir_test: Directory containing images or single image path.

    Returns:
        test_images: List containing all image paths.
    """
    test_images = []
    if os.path.isdir(dir_test):
        image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
        for file_type in image_file_types:
            test_images.extend(glob.glob(f"{dir_test}/{file_type}"))
    else:
        test_images.append(dir_test)
    return test_images    


# Select the experiment-name, model from the list of experiments
def select_experiment(experiments, experiment_name):
    for experiment in experiments:
        if experiment.name == experiment_name:
            return experiment
    return None



def load_model_from_Mlflow(model_name, model_stage):
    client = MlflowClient()
    model_info = client.get_latest_versions(model_name, stages=[model_stage])[0] 
    run_id = model_info.run_id

    # Get all experiments
    experiments = mlflow.list_experiments()

    # Print the list of experiments
   # print("Available experiments:")
    #for experiment in experiments:
     #   print(experiment.name)
    # Select the desired experiment

    experiment_name = 'fasterrcnn_exp'
    selected_experiment = select_experiment(experiments, experiment_name)

    if selected_experiment is not None:
        # Do something with the selected experiment
        print(f"Selected experiment: {selected_experiment.name}")
    else:
        print(f"Experiment '{experiment_name}' not found.")
        
    artifact_uri = mlflow.get_run(run_id).info.artifact_uri
    model_state = mlflow.pytorch.load_state_dict(artifact_uri + "/model_state")
    model_uri = "models:/{model_name}/{model_stage}".format(model_name=model_name, model_stage=model_stage)
    checkpoint = mlflow.pytorch.load_model(model_uri)
    return model_state, checkpoint

    # Change model transition to another stage

    # mlflow_run.transition_model_version_stage(
    # name=model_name,
    # stage="Staging",)

def main(args,model_name, model_stage):
    # For same annotation colors each time.
    np.random.seed(42)
        
    if  args['device'] and torch.cuda.is_available():
         DEVICE  = torch.device('cuda:0')
    else:
         DEVICE  = torch.device('cpu')
    print(f'Device: {DEVICE}')

    # client = MlflowClient()
    
    # model_info = client.get_latest_versions(model_name, stages=[model_stage])[0] 
    # run_id = model_info.run_id
    # artifact_uri = mlflow.get_run(run_id).info.artifact_uri
    # model_state=mlflow.pytorch.load_state_dict(artifact_uri + "/model_state")
    # model_uri = "models:/{model_name}/{model_stage}".format(model_name=model_name, model_stage=model_stage)
    # checkpoint = mlflow.pytorch.load_model(model_uri)
  
    # Load model from MLflow
    model_state, checkpoint = load_model_from_Mlflow(model_name, model_stage)

    NUM_CLASSES =len(model_state['data']['CLASSES'])
    print('num_class', NUM_CLASSES)
    CLASSES=(model_state['data']['CLASSES'])   
    build_model = create_model[model_state['model_name']]  
    model = build_model(num_classes=NUM_CLASSES, coco_model=False)
    model.load_state_dict(checkpoint.state_dict())
    model.to(DEVICE).eval()
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    #print('DIR_TEST')
    test_images = args['input']
     
    print(f"Test instances: {len(test_images)}")

    # Define the detection threshold any detection having
    # score below this will be discarded.
    detection_threshold = args['threshold']

    # To count the total number of frames iterated through.
    frame_count = 0
    # To keep adding the frames' FPS.
    total_fps = 0
    for i in range(len(test_images)):
        # Get the image file name for saving output later on.
        image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
        
        orig_image = cv2.imread(test_images[i])
      
        frame_height, frame_width, _ = orig_image.shape
       
        if args['imgsz'] != None:
            RESIZE_TO = args['imgsz']
        else:
            RESIZE_TO = frame_width
        # orig_image = image.copy()
        image_resized = resize(
            orig_image, RESIZE_TO, square=args['square_img']
        )
        image = image_resized.copy()
        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = infer_transforms(image)
       
        # Add batch dimension.
        image = torch.unsqueeze(image, 0)
        start_time = time.time()

        with torch.no_grad():
            outputs = model(image.to(DEVICE))
        end_time = time.time()
   
        # Get the current fps.
        fps = 1 / (end_time - start_time)
        # Add `fps` to `total_fps`.
        total_fps += fps
        # Increment frame count.
        frame_count += 1
        # Load all detection to CPU for further operations.
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        # Carry further only if there are detected boxes.
        if len(outputs[0]['boxes']) != 0:
            orig_image = inference_annotations(
                outputs, 
                detection_threshold, 
                CLASSES,
                COLORS, 
                orig_image, 
                image_resized,
                args
            )
        is_success, buffer = cv2.imencode('.png', orig_image)
      
        io_buf = BytesIO(buffer)
        io_buf.seek(0)
        print(f"Image {i+1} done...")
        print('-'*50)

    print('TEST PREDICTIONS COMPLETE')
    # Calculate and print the average FPS.
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")
    for item in outputs:
        item['boxes'] = item['boxes'].tolist()
        item['labels'] = item['labels'].tolist()
        item['scores'] = item['scores'].tolist()
    json_string = json.dumps(outputs)
    return  json_string , io_buf

if __name__ == '__main__':
    args={'input': ['/home/ubuntu/data/fasterrcnn/fasterrcnn_pytorch_api/data/submarine_det/train/img/2019-02-20_19-01-02to2019-02-20_19-01-13_1-0001_jpg.rf.168dadf99e426cf045bc56ecf377eaba.jpg'],
           'accept':'image/png',
           'square_img':False,
           'no_labels':False,
           'device':False,
           'imgsz':640,
           'threshold':0.7

           }
    model_name = 'fasterrcnn'
    model_stage = 'Production'
    main(args, model_name, model_stage)