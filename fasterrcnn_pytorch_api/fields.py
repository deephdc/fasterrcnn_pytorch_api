import json
from webargs import fields, validate
from marshmallow import Schema, fields
from fasterrcnn_pytorch_api import configs
 
 
#####################################################
#  Options to  train your model
#####################################################

class AugTrainingOptionSchema(Schema):
    blur = fields.Dict(required=True, allow_none=False, keys=fields.Str(), values=fields.Float())
    motion_blur = fields.Dict(required=True, allow_none=False, keys=fields.Str(), values=fields.Float())
    median_blur = fields.Dict(required=True, allow_none=False, keys=fields.Str(), values=fields.Float())
    to_gray = fields.Dict(required=True, allow_none=False, keys=fields.Str(), values=fields.Float())
    random_brightness_contrast = fields.Dict(required=True, allow_none=False, keys=fields.Str(), values=fields.Float())
    color_jitter = fields.Dict(required=True, allow_none=False, keys=fields.Str(), values=fields.Float())
    random_gamma = fields.Dict(required=True, allow_none=False, keys=fields.Str(), values=fields.Float())
    horizontal_flip = fields.Dict(required=True, allow_none=False, keys=fields.Str(), values=fields.Int())
    vertical_flip = fields.Dict(required=True, allow_none=False, keys=fields.Str(), values=fields.Int())
    rotate = fields.Dict(required=True, allow_none=False, keys=fields.Str(), values=fields.Int())
    shift_scale_rotate = fields.Dict(required=True, allow_none=False, keys=fields.Str(), values=fields.Float())
    Cutout = fields.Dict(required=True, allow_none=False, keys=fields.Str(), values=fields.Int())
    ChannelShuffle = fields.Dict(required=True, allow_none=False, keys=fields.Str(), values=fields.Int())
 
     
    

class TrainArgsSchema(Schema):

    class Meta:
        ordered = True

    model= fields.Str(
        enum= configs.BACKBONES,
        required=True,
        description= 'name of the model.' )

    data_config= fields.Str(
        required=True,
        description= 'path to the data config file.')
    
    use_train_aug= fields.Bool(#FIXME: give choice to user which transforms they want to use
        required=False,
        missing=False,
        enum=[True,False],
        description='whether to use train augmentation, uses some advanced augmentation' 
                     'that may make training difficult when used with mosaic. If true, it use the options'\
                     'in aug_training_option. You can change that to have costum augumentation.' )
    #aug_training_option = fields.Str( missing="{'blur': 0.1}")
    #aug_training_option = fields.Nested(AugTrainingOptionSchema, required=False, allow_none=True)
    aug_training_option = fields.Str(
        required=False,
        missing=str(configs.DATA_AGU_OPTION),
        description='augmentation options.\n'
                    'blur_limit (int) - maximum kernel size for blurring the input image.\n'
                    'p (float) - probability of applying the transform.\n'
                    'shift_limit ((float, float) or float) - shift factor range for both height and width.\n'
                    'scale_limit ((float, float) or float) - scaling factor range.\n'
                    'rotate_limit ((int, int) or int) - rotation range.\n'
                    'num_holes (int) - number of regions to zero out.\n'
                    'max_h_size (int) - maximum height of the hole.\n'
                    'max_w_size (int) - maximum width of the hole.\n'
    )


    eval_n_epochs= fields.Int(
        required=False,
        missing=1,
        description= 'Evalute the model every n epochs during training.')
    
    device= fields.Bool(
        required=False,
        missing=True,
        enum=[True,False],
        description= 'computation/training device, default is GPU if GPU present.')
    
    epochs= fields.Int(
        required=False,
        missing=4,
        description= 'number of epochs to train.' )

    workers= fields.Int(
        required=False,
        missing=4,
        description= 'number of workers for data processing/transforms/augmentations.')

    batch= fields.Int(
        required=False,
        missing=4,
        description= 'batch size to load the data.')

    lr= fields.Float(
        required=False,
        missing=0.001,
        description= 'batch size to load the data.')

    imgsz= fields.Int(
        required=False,
        missing=640,
        description= 'image size to feed to the network.')     
    
    no_mosaic= fields.Bool(
        required=False,
        missing=True,
        enum=[True,False],
        description= 'pass this to not to use mosaic augmentation.')

    cosine_annealing= fields.Bool(
        required=False,
        missing=True,
        enum=[True,False],
        description= 'use cosine annealing warm restarts.')

    weights= fields.Str(
        required=False,
        missing=None,
        description= 'path to model weights if using custome pretrained weights.'\
            'The name of the directory that contains the checkpint within the "model" directory.')

    resume_training= fields.Bool(
        required=False,
        missing=False,
        enum=[True,False],
        description= 'If using custome pretrained weights, resume trining from the last step of the provided checkpoint.'\
            'If True, the path to the weights should be specified in the argument weights')

    square_training= fields.Bool(
        required=False,
        missing=True,
        enum=[True,False],
        description= ' Resize images to square shape instead of aspect ratio resizing for single image training.'
                    'For mosaic training, this resizes single images to square shape first then puts them on a square canvas.')
    
    seed= fields.Int(
        required=False,
        missing=0,
        description= 'golabl seed for training.')
    
#####################################################
#  Options to test your model
#####################################################

class PredictArgsSchema(Schema):
    
    class Meta:
        ordered = True

    input= fields.Field(
        required=True,
        type="file",
        location="form",
        description= 'Input an image.')    

    timestamp= fields.Str(
        required=False,
        description= 'Model timestamp to be used for prediction. To see the available timestamp,'
              ' please run the get_metadata function. If no timestamp is given,'
              ' the model will be loaded from coco will be loaded.')
    
    model= fields.Str(
        required=False,
        enum= configs.BACKBONES,
        description= 'Name of the model you want to use for inference.')

    threshold= fields.Float(
        required=False,
        missing=0.5,
        description= 'detection threshold.')

    imgsz= fields.Int(
        required=False,
        missing=640,
        description= 'image size to feed to the network.')
       
    device= fields.Bool(
        required=False,
        missing=True,
        enum=[True,False],
        description= ' computation device, default is GPU if GPU present.')
    
    no_labels= fields.Bool(
        required=False,
        missing=False,
        enum=[True,False],
        description= 'visualize output only if this argument is passed.')
    
    square_img= fields.Bool(
        required=False,
        missing=True,
        enum=[True,False],
        description= 'whether to use square image resize, else use aspect ratio resize.')

    accept= fields.Str(
        missing ="application/pdf",
        location="headers",
        validate =validate.OneOf(['image/png', 'application/json']) ,
        description ="Returns png file with detection resutlts or a json with the prediction.")
 
if __name__=='__main__':
   pass    
