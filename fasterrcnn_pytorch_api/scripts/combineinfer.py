import numpy as np
import cv2
import torch
import time
import yaml
import os
import json
from io import BytesIO
import tempfile
import mimetypes
from fasterrcnn_pytorch_training_pipeline.models.create_fasterrcnn_model import create_model
from fasterrcnn_pytorch_training_pipeline.utils.annotations import inference_annotations, annotate_fps
from fasterrcnn_pytorch_training_pipeline.utils.transforms import infer_transforms, resize
from fasterrcnn_pytorch_api import configs

def get_video_dimensions(video_path):
    """
    Reads a video and returns the capture object and its frame dimensions.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        cv2.VideoCapture: Video capture object.
        int: Frame width.
        int: Frame height.
    """
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    assert frame_width != 0 and frame_height != 0, 'Please check video path...'
    return cap, frame_width, frame_height

class InferenceEngine:
    def __init__(self, args):
        self.device = torch.device('cuda:0' if args['device'] and torch.cuda.is_available() else 'cpu')
        self.build_model(args)

    def build_model(self, args):
        """
        Builds the model for inference based on the given arguments.

        Args:
            args (dict): Dictionary of arguments.

        Returns:
            None
        """
        if args['weights'] is None:
            with open(os.path.join(configs.DATA_PATH, 'coco_config/coco_config.yaml')) as file:
                data_configs = yaml.safe_load(file)
            NUM_CLASSES = data_configs['NC']
            self.CLASSES = data_configs['CLASSES']
            try:
                build_model_fn = create_model[args['model']]
                self.model, _ = build_model_fn(num_classes=NUM_CLASSES, coco_model=True)
            except KeyError:
                build_model_fn = create_model['fasterrcnn_resnet50_fpn_v2']
                self.model, _ = build_model_fn(num_classes=NUM_CLASSES, coco_model=True)
            self.colors = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))
        else:
            checkpoint = torch.load(args['weights'], map_location=self.device)
            NUM_CLASSES = len(checkpoint['data']['CLASSES'])
            self.CLASSES = checkpoint['data']['CLASSES']
            self.colors = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))
            build_model_fn = create_model[checkpoint['model_name']]
            self.model = build_model_fn(num_classes=NUM_CLASSES, coco_model=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device).eval()

    def infer_video(self, video_path, **args):
        """
        Performs inference on a video and returns JSON results and a video message.

        Args:
            video_path (str): Path to the input video file.
            **args: Additional arguments.

        Returns:
            dict: JSON output containing annotations for each frame.
            binary: Binary video message.
        """
        cap, frame_width, frame_height = get_video_dimensions(video_path)
        output_format = 'mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        with tempfile.NamedTemporaryFile(suffix='.' + output_format, delete=False) as temp_file:
            temp_filename = temp_file.name
            out = cv2.VideoWriter(temp_filename, fourcc, 30, (frame_width, frame_height))
            json_out = {}
            frame_count = 0
            while cap.isOpened():
                print(f"processing frame {frame_count}")
                ret, frame = cap.read()
                if not ret:
                    break

                orig_frame = frame.copy()
                orig_frame, json_string, fps = self.generate_json_response(orig_frame, frame_width, **args)
                orig_frame = annotate_fps(orig_frame, fps)
                out.write(orig_frame)
                json_out[f'frame_{frame_count}'] = json.loads(json_string)
                frame_count += 1

        cap.release()
        out.release()
        final_filename = 'output.mp4'
        os.rename(temp_filename, final_filename)
        message = open(final_filename, 'rb')
        return json_out, message

    def infer_single_image(self, image_path, **args):
        """
        Performs inference on a single image and returns JSON results and image buffer.

        Args:
            image_path (str): Path to the input image file.
            **args: Additional arguments.

        Returns:
            str: JSON output containing annotations.
            io.BytesIO: Binary image buffer.
        """
        orig_image = cv2.imread(image_path)
        frame_height, frame_width, _ = orig_image.shape

        orig_image, json_string, fps = self.generate_json_response(orig_image, frame_width, **args)
        is_success, buffer = cv2.imencode('.png', orig_image)

        io_buf = BytesIO(buffer)
        io_buf.seek(0)
        print('-' * 50)
        print('TEST PREDICTIONS COMPLETE')
        print(f"FPS: {fps:.3f}")
        return json_string, io_buf

    def generate_json_response(self, orig_image, frame_width, **args):
        """
        Generates JSON response and annotations based on the inference results.

        Args:
            orig_image (numpy.ndarray): Original image data.
            frame_width (int): Frame width.
            **args: Additional arguments.

        Returns:
            numpy.ndarray: Image with annotations.
            str: JSON output containing annotations.
            float: Frames per second (FPS) value.
        """
        if args['imgsz'] is not None:
            RESIZE_TO = args['imgsz']
        else:
            RESIZE_TO = frame_width
        image_resized = resize(orig_image, RESIZE_TO, square=args['square_img'])
        image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image = infer_transforms(image)
        image = torch.unsqueeze(image, 0)
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(image.to(self.device))
        end_time = time.time()
        fps = 1 / (end_time - start_time)

        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

        if len(outputs[0]['boxes']) != 0:
            orig_image = inference_annotations(
                outputs, args['threshold'], self.CLASSES, self.colors, orig_image, image_resized, args
            )
        for item in outputs:
            item['boxes'] = item['boxes'].tolist()
            item['labels'] = item['labels'].tolist()
            item['scores'] = item['scores'].tolist()
        json_string = json.dumps(outputs)

        return orig_image, json_string, fps

    def infer(self, file_format, **args):
        """
        Perform inference on given inputs and return results.

        Args:
            file_format (str): Type of input ('video' or 'image').
            **args: Additional arguments.

        Returns:
            tuple: Tuple containing JSON results and binary message.
        """
        test_inputs = args['input']
        results = []
        for input in test_inputs:
            if file_format == 'video':
                return self.infer_video(input, **args)
            elif file_format == 'image':
                return self.infer_single_image(input, **args)
            else:
                print('Please provide a valid input type')

if __name__ == '__main__':
    args = {
        'device': 'cuda',  # 'cuda' or 'cpu'
        'weights': 'path_to_weights.pth',  # Provide the path to your weights file
        'model': 'your_model_name',  # Specify your model name
        'imgsz': None,  # Specify your desired image size
        'square_img': False,  # Specify whether to resize to square images
        'threshold': 0.3  # Specify your detection threshold
    }
    engine = InferenceEngine(args)
    file_format = 'video'  # Specify input type ('video' or 'image')
    results = engine.infer(file_format, **args)
    print(results)
