from io import BytesIO
import json
import numpy as np
import cv2
import torch
import glob as glob
import os
import time

import yaml

from fasterrcnn_pytorch_training_pipeline.models.create_fasterrcnn_model import create_model
from fasterrcnn_pytorch_training_pipeline.utils.annotations import inference_annotations
from fasterrcnn_pytorch_training_pipeline.utils.transforms import infer_transforms, resize
from fasterrcnn_pytorch_api import configs

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


def main(args):
    # For same annotation colors each time.
    np.random.seed(42)
        
    if  args['device'] and torch.cuda.is_available():
         DEVICE  = torch.device('cuda:0')
    else:
         DEVICE  = torch.device('cpu')
    print(f'Device: {DEVICE}')
    
    # Load the pretrained model
    if args['weights'] is None:
        #load the default one for COCO.
        
        with open(os.path.join(configs.DATA_PATH, 'coco_config.yaml')) as file:
                data_configs = yaml.safe_load(file)
        NUM_CLASSES = data_configs['NC']
        CLASSES = data_configs['CLASSES']
        try:
            build_model = create_model[args['model']]
            model, _ = build_model(num_classes=NUM_CLASSES, coco_model=True)
        except:
            build_model = create_model['fasterrcnn_resnet50_fpn_v2']
            model, _ = build_model(num_classes=NUM_CLASSES, coco_model=True)
    else:        
        checkpoint = torch.load(args['weights'], map_location=DEVICE)
        NUM_CLASSES =len(checkpoint['data']['CLASSES'])
        CLASSES=(checkpoint['data']['CLASSES'])   
        build_model = create_model[checkpoint['model_name']]  
        model = build_model(num_classes=NUM_CLASSES, coco_model=False)
        model.load_state_dict(checkpoint['model_state_dict'])
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
    from fasterrcnn_pytorch_api import configs
    weights=os.path.join(configs.MODEL_DIR,  '2023-05-10_121810', 'last_model.pth')
    print(weights)
    checkpoint = torch.load(weights, map_location='cpu')
    NUM_CLASSES =len(checkpoint['data']['CLASSES'])
    print('NUM_CLASS', NUM_CLASSES)
    CLASSES=(checkpoint['data']['CLASSES'])   
    build_model = create_model[checkpoint['model_name']]  
    model = build_model(num_classes=NUM_CLASSES, coco_model=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to('cpu').eval()
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))