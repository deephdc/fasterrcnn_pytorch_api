from os import path
from webargs import fields, validate
 

#####################################################
#  GENERAL CONFIG
#####################################################
BASE_DIR=path.dirname(path.normpath(path.dirname(__file__)))
DATASET_DIR = path.join(BASE_DIR,'data/submarine_det') # Location of the dataset
MODEL_DIR=path.join(BASE_DIR,'models')


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
 description= ' path to model weights if using pretrained weights.'
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
'data_config': fields.Str(
required=False,
 description= 'path to the data config file.'
 ),
'model': fields.Str(
required=True,
 description= 'name of the model.'
 ),
 
'timestamp': fields.Str(
required=True,
 description= 'Model timestamp to be used for prediction. To see the available timestamp for each header task, please run the get_metadata function.'
 ), 
'weights' :fields.Str(
required=True,
 description= ' path to trained checkpoint weights.'),

'threshold' :fields.Float(
required=True,
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
missing=True,
enum=[True,False],
 description= 'visualize output only if this argument is passed.'
 ),
'square_img': fields.Bool(
required=False,
missing=True,
enum=[True,False],
 description= 'whether to use square image resize, else use aspect ratio resize.'
 ),

 }

  
      
 
  
 
 
 
                     