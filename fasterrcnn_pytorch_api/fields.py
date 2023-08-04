import json
from webargs import fields, validate
from marshmallow import Schema, ValidationError, fields
from fasterrcnn_pytorch_api import configs

#####################################################
#  Options to  train your model
#####################################################
class MyCustomFieldForJson(fields.String):
    def __init__(self, *args, **kwargs):
        self.metadata = kwargs.get('metadata', {})
        self.metadata['description'] = kwargs.get('description')
        super().__init__(*args, **kwargs)
    def _deserialize(self, value, attr, data, **kwargs):
        try:
            return json.loads(value)
        except json.JSONDecodeError as err:
            raise ValidationError(f"Invalid JSON: `{err}`")
        
    def _validate(self, value):
        if not isinstance(value, dict):
            raise ValidationError("Invalid value. Expected a dictionary.")
       
        for k1, v1 in value.items():
            if not isinstance(v1, dict):
                raise ValidationError(f"Invalid value for {k1}. Expected a dictionary.")

            for k2 , v2 in v1.items():
                if k2 == 'p':
                    if not isinstance(v2,float ) or not (0 <= v2 <= 1.0):
                        raise ValidationError(f"Invalid value for 'p' in {k2}: {v2} It must be a float or integer between 0 and 1.")
                elif k2 in ['max_w_size', 'max_h_size','num_holes', 'blur_limit']:
                    if not isinstance(v2, int) or v2 < 0:
                        raise ValidationError(f"Invalid value for  '{k2}' in {k1}: {v2}. It must be a non-negative integer.")
                elif k2  in  ['scale_limit', 'shift_limit' ]:
                    if not isinstance(v2, float) or not isinstance(v2,(float, float)):
                        raise ValidationError(f"Invalid value for '{k2}': {v2}. It must be a float or (float,float).")
                elif k2=='rotate_limit':
                    if not isinstance(v2, int) or not isinstance(v2,(int, int)):
                        raise ValidationError(f"Invalid value for '{k2}' : {v2}. It must be a int or (int, int).")

                        
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
  
    aug_training_option = MyCustomFieldForJson(
        description='augmentation options.\n'
                    'blur_limit (int) - maximum kernel size for blurring the input image.\n'
                    'p (float) - probability of applying the transform.\n'
                    'shift_limit ((float, float) or float) - shift factor range for both height and width.\n'
                    'scale_limit ((float, float) or float) - scaling factor range.\n'
                    'rotate_limit ((int, int) or int) - rotation range.\n'
                    'num_holes (int) - number of regions to zero out.\n'
                    'max_h_size (int) - maximum height of the hole.\n'
                    'max_w_size (int) - maximum width of the hole.\n',
        missing=json.dumps(configs.DATA_AUG_OPTION)

    )
    
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
    disable_wandb=fields.Bool(
        required=False,
        missing=True,
        description= 'whether to use the wandb'
    )
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
        missing=None,
        description= 'Model timestamp to be used for prediction. To see the available timestamp,'
              ' please run the get_metadata function. If no timestamp is given,'
              ' the model will be loaded from coco will be loaded.')
    
    model= fields.Str(
        required=False,
        missing="fasterrcnn_resnet50_fpn_v2",
        enum= configs.BACKBONES,
        description= 'Please provide the name of the model you want to use for inference.'
                      'If you have specified neither nethier timestamp nor model name,'
                      'the default model "fasterrcnn_resnet50_fpn_v2" is loaded.')

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
