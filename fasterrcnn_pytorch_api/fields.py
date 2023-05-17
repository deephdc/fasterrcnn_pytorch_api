from os import path
from webargs import fields, validate
 
 
#####################################################
#  Options to  train your model
#####################################################

training_args={'model': fields.Str(
   enum= [
    'fasterrcnn_convnext_small',
    'fasterrcnn_convnext_tiny',
    'fasterrcnn_custom_resnet', 
    'fasterrcnn_darknet',
    'fasterrcnn_efficientnet_b0',
    'fasterrcnn_efficientnet_b4',
    'fasterrcnn_mbv3_small_nano_head',
    'fasterrcnn_mbv3_large',
    'fasterrcnn_mini_darknet_nano_head',
    'fasterrcnn_mini_darknet',
    'fasterrcnn_mini_squeezenet1_1_small_head',
    'fasterrcnn_mini_squeezenet1_1_tiny_head',
    'fasterrcnn_mobilenetv3_large_320_fpn',
    'fasterrcnn_mobilenetv3_large_fpn',
    'fasterrcnn_nano',
    'fasterrcnn_resnet18',
    'fasterrcnn_resnet50_fpn_v2',
    'fasterrcnn_resnet50_fpn', 
    'fasterrcnn_resnet101',
    'fasterrcnn_resnet152',
    'fasterrcnn_squeezenet1_0',
    'fasterrcnn_squeezenet1_1_small_head',
    'fasterrcnn_squeezenet1_1',
    'fasterrcnn_vitdet',
    'fasterrcnn_vitdet_tiny',
    'fasterrcnn_mobilevit_xxs',
    'fasterrcnn_regnet_y_400mf'
],
required=True,
description= 'name of the model.'
 ),

'data_config': fields.Str(
required=False,
description= 'path to the data config file.'
 ),
'use_train_aug': fields.Bool(
required=False,
missing=False,
enum=[True,False],
description= 'whether to use train augmentation, uses some advanced augmentation that may make training    difficult when used with mosaic.'
 ),

'device': fields.Bool(
required=False,
missing=True,
enum=[True,False],
 description= ' computation/training device, default is GPU if GPU present.'
 ),
'epochs': fields.Int(
required=False,
missing=4,
 description= 'number of epochs to train.'
 ),

'workers': fields.Int(
required=False,
missing=4,
 description= ' number of workers for data processing/transforms/augmentations.'
 ),

'batch': fields.Int(
required=False,
missing=4,
 description= 'batch size to load the data.'
 ),

'lr': fields.Float(
required=False,
missing=0.001,
 description= 'batch size to load the data.'
 ),

'imgsz': fields.Int(
required=False,
missing=640,
 description= 'image size to feed to the network.'
 ),        
          
 
'vis_transformed': fields.Bool(
required=False,
missing=True,
enum=[True,False],
 description= 'visualize transformed images fed to the network '
 ),
'no_mosaic': fields.Bool(
required=False,
missing=True,
enum=[True,False],
 description= ' pass this to not to use mosaic augmentation.'
 ), 

'use_train_aug': fields.Bool(
required=False,
missing=True,
enum=[True,False],
 description= ' whether to use train augmentation, uses some advanced augmentation that may make training difficult when used with mosaic.'
 ), 

'SAVE_VALID_PREDICTION_IMAGES': fields.Bool(
required=False,
missing=True,
enum=[True,False],
 description= 'Whether to save the predictions of the validation set while training.'
 ),

'cosine_annealing': fields.Bool(
required=False,
missing=True,
enum=[True,False],
 description= 'use cosine annealing warm restarts.'
 ),

'weights': fields.Str(
required=False,
missing=None,
 description= 'path to model weights if using pretrained weights.'
 ),

'resume_training': fields.Bool(
required=False,
missing=False,
enum=[True,False],
 description= 'If using pretrained weights, resume trining from the last step of the provided checkpoint .'
 ),

'square_training': fields.Bool(
required=False,
missing=True,
enum=[True,False],
 description= ' Resize images to square shape instead of aspect ratio resizing for single image training. For mosaic training, this resizes single images to square shape first then puts them on a square canvas.'
 ),

'seed': fields.Int(
required=False,
missing=0,
 description= 'golabl seed for training. '
 ),

 }
     
#####################################################
#  Options to test your model
#####################################################

predict_args={
'input': fields.Field(
required=True,
type="file",
location="form",
 description= 'Input an image.'
 ),    


'timestamp': fields.Str(
required=True,
 description= 'Model timestamp to be used for prediction. To see the available timestamp for each header task, please run the get_metadata function.'
 ), 


'threshold' :fields.Float(
required=False,
missing=0.5,
 description= 'detection threshold.'),

'imgsz': fields.Int(
required=False,
missing=640,
 description= 'image size to feed to the network.'
 ),        
'device': fields.Bool(
required=False,
missing=True,
enum=[True,False],
 description= ' computation device, default is GPU if GPU present.'
 ), 
'no_labels': fields.Bool(
required=False,
missing=False,
enum=[True,False],
 description= 'visualize output only if this argument is passed.'
 ),
'square_img': fields.Bool(
required=False,
missing=True,
enum=[True,False],
 description= 'whether to use square image resize, else use aspect ratio resize.'
 ),

'accept':fields.Str(
                    missing ="application/pdf",
                    location="headers",
                    validate =validate.OneOf(['image/png', 'application/json']) ,
                    description ="Returns png file with detection resutlts or a json with the prediction.")
 }
 
'''
class ModelName(fields.String):
    """Name of the model."""
    def __init__(self, **kwargs):
        super().__init__(
            enum=[
                'fasterrcnn_convnext_small',
                'fasterrcnn_convnext_tiny',
                'fasterrcnn_custom_resnet', 
                'fasterrcnn_darknet',
                'fasterrcnn_efficientnet_b0',
                'fasterrcnn_efficientnet_b4',
                'fasterrcnn_mbv3_small_nano_head',
                'fasterrcnn_mbv3_large',
                'fasterrcnn_mini_darknet_nano_head',
                'fasterrcnn_mini_darknet',
                'fasterrcnn_mini_squeezenet1_1_small_head',
                'fasterrcnn_mini_squeezenet1_1_tiny_head',
                'fasterrcnn_mobilenetv3_large_320_fpn',
                'fasterrcnn_mobilenetv3_large_fpn',
                'fasterrcnn_nano',
                'fasterrcnn_resnet18',
                'fasterrcnn_resnet50_fpn_v2',
                'fasterrcnn_resnet50_fpn', 
                'fasterrcnn_resnet101',
                'fasterrcnn_resnet152',
                'fasterrcnn_squeezenet1_0',
                'fasterrcnn_squeezenet1_1_small_head',
                'fasterrcnn_squeezenet1_1',
                'fasterrcnn_vitdet',
                'fasterrcnn_vitdet_tiny',
                'fasterrcnn_mobilevit_xxs',
                'fasterrcnn_regnet_y_400mf'
            ],
            required=True,
            description=self.__doc__,
            **kwargs
        )

class DataConfigPath(fields.String):
    """Path to the data config file."""
    def __init__(self, **kwargs):
        super().__init__(required=False, description=self.__doc__, **kwargs)

class UseTrainAugmentation(fields.Boolean):
    """Whether to use train augmentation, uses some advanced augmentation that may make training difficult when used with mosaic."""
    def __init__(self, **kwargs):
        super().__init__(
            required=False,
            missing=False,
            enum=[True, False],
            description=self.__doc__,
            **kwargs
        )

class DeviceType(fields.Boolean):
    """Computation/training device, default is GPU if GPU present."""
    def __init__(self, **kwargs):
        super().__init__(
            required=False,
            missing=True,
            enum=[True, False],
            description=self.__doc__,
            **kwargs
        )

class NumberOfEpochs(fields.Integer):
    """Number of epochs to train."""
    def __init__(self, **kwargs):
        super().__init__(
            required=False,
            missing=4,
            description=self.__doc__,
            **kwargs
        )

class NumberOfWorkers(fields.Integer):
    """Number of workers for data processing/transforms/augmentations."""
    def __init__(self, **kwargs):
        super().__init__(
            required=False,
            missing=4,
            description=self.__doc__,
            **kwargs
        )

class BatchSize(fields.Integer):
    """Batch size to load the data."""
    def __init__(self, **kwargs):
        super().__init__(
            required=False,
            missing=4,
            description=self.__doc__,
            **kwargs
        )

class LearningRate(fields.Float):
    """Learning rate for training."""
    def __init__(self, **kwargs):
        super().__init__(required=False, missing=0.001, description=self.__doc__, **kwargs)
        
        
class ImageSize(fields.Integer):
    """Image size to feed to the network."""
    def __init__(self, **kwargs):
        super().__init__(required=False, missing=640, description=self.__doc__, **kwargs)
        
        
class VisualizeTransformed(fields.Boolean):
    """Visualize transformed images fed to the network."""
    def __init__(self, **kwargs):
        super().__init__(required=False, missing=True, enum=[True,False], description=self.__doc__, **kwargs)
        
        
class NoMosaic(fields.Boolean):
    """Do not use mosaic augmentation."""
    def __init__(self, **kwargs):
        super().__init__(required=False, missing=True, enum=[True,False], description=self.__doc__, **kwargs)
        
        
class UseTrainAug(fields.Boolean):
    """Whether to use train augmentation, uses some advanced augmentation that may make training difficult when used with mosaic."""
    def __init__(self, **kwargs):
        super().__init__(required=False, missing=True, enum=[True,False], description=self.__doc__, **kwargs)
        
        
class SaveValidPredImgs(fields.Boolean):
    """Whether to save the predictions of the validation set while training."""
    def __init__(self, **kwargs):
        super().__init__(required=False, missing=True, enum=[True,False], description=self.__doc__, **kwargs)
        
        
class CosineAnnealing(fields.Boolean):
    """Use cosine annealing warm restarts."""
    def __init__(self, **kwargs):
        super().__init__(required=False, missing=True, enum=[True,False], description=self.__doc__, **kwargs)
        
        
class ModelWeights(fields.String):
    """Path to model weights if using pretrained weights."""
    def __init__(self, **kwargs):
        super().__init__(required=False, missing=None, description=self.__doc__, **kwargs)
        
        
class ResumeTraining(fields.Boolean):
    """Path to model weights if using pretrained weights."""
    def __init__(self, **kwargs):
        super().__init__(required=False, missing=False, enum=[True,False], description=self.__doc__, **kwargs)
        
        
class SquareTraining(fields.Boolean):
    """Resize images to square shape instead of aspect ratio resizing for single image training. For mosaic training, this resizes single images to square shape first then puts them on a square canvas."""
    def __init__(self, **kwargs):
        super().__init__(required=False, missing=True, enum=[True,False], description=self.__doc__, **kwargs)
        
        
class GlobalSeed(fields.Integer):
    """Global seed for training."""
    def __init__(self, **kwargs):
        super().__init__(required=False, missing=0, description=self.__doc__, **kwargs)


class InputImage(fields.Field):
    """
    Input an image.
    """
    def __init__(self, **kwargs):
        super().__init__(
            required=True, type="file", location="form", description=self.__doc__, **kwargs
        )

class ModelTimestamp(fields.Str):
    """
    Model timestamp to be used for prediction. To see the available timestamp for each header task, please run the get_metadata function.
    """
    def __init__(self, **kwargs):
        super().__init__(required=True, description=self.__doc__, **kwargs)

class CheckpointWeights(fields.Str):
    """
    Path to trained checkpoint weights.
    """
    def __init__(self, **kwargs):
        super().__init__(required=False, missing="best_model.pth", description=self.__doc__, **kwargs)

class DetectionThreshold(fields.Float):
    """
    Detection threshold.
    """
    def __init__(self, **kwargs):
        super().__init__(required=False, missing=0.5, description=self.__doc__, **kwargs)

class ImageSize(fields.Int):
    """
    Image size to feed to the network.
    """
    def __init__(self, **kwargs):
        super().__init__(required=False, missing=640, description=self.__doc__, **kwargs)

class ComputationDevice(fields.Bool):
    """
    Computation device, default is GPU if GPU present.
    """
    def __init__(self, **kwargs):
        super().__init__(required=False, missing=True, enum=[True,False], description=self.__doc__, **kwargs)

class VisualizeOutputOnly(fields.Bool):
    """
    Visualize output only if this argument is passed.
    """
    def __init__(self, **kwargs):
        super().__init__(required=False, missing=True, enum=[True,False], description=self.__doc__, **kwargs)

class SquareImageResize(fields.Bool):
    """
    Whether to use square image resize, else use aspect ratio resize.
    """
    def __init__(self, **kwargs):
        super().__init__(required=False, missing=True, enum=[True,False], description=self.__doc__, **kwargs)

class AcceptHeader(fields.Str):
    """
    Returns png file with detection results or a json with the prediction.
    """
    def __init__(self, **kwargs):
        super().__init__(missing="application/pdf", location="headers", validate=validate.OneOf(['image/png', 'application/json']), description=self.__doc__, **kwargs)

# Define predict_args object
predict_args = {
    'input': InputImage(),
    'timestamp': ModelTimestamp(),
    'weights': CheckpointWeights(),
    'threshold': DetectionThreshold(),
    'imgsz': ImageSize(),
    'device': ComputationDevice(),
    'no_labels': VisualizeOutputOnly(),
    'square_img': SquareImageResize(),
    'accept': AcceptHeader()
}
                
'''
                     