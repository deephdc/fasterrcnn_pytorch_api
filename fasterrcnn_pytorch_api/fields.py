from webargs import fields, validate
from marshmallow import Schema, fields

 
 
#####################################################
#  Options to  train your model
#####################################################
class TrainArgsSchema(Schema):

    class Meta:
        ordered = True

    model= fields.Str(
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
        description= 'name of the model.' )

    data_config= fields.Str(
        required=False,
        description= 'path to the data config file.')
    
    use_train_aug= fields.Bool(
        required=False,
        missing=False,
        enum=[True,False],
        description='whether to use train augmentation, uses some advanced augmentation' 
                     'that may make training difficult when used with mosaic.' )

    device= fields.Bool(
        required=False,
        missing=True,
        enum=[True,False],
        description= ' computation/training device, default is GPU if GPU present.')
    
    epochs= fields.Int(
        required=False,
        missing=4,
        description= 'number of epochs to train.' )

    workers= fields.Int(
        required=False,
        missing=4,
        description= ' number of workers for data processing/transforms/augmentations.')

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
        description= ' pass this to not to use mosaic augmentation.')

    use_train_aug= fields.Bool(
        required=False,
        missing=True,
        enum=[True,False],
        description= ' whether to use train augmentation, uses some advanced augmentation'\
                     'that may make training difficult when used with mosaic.')

    cosine_annealing= fields.Bool(
        required=False,
        missing=True,
        enum=[True,False],
        description= 'use cosine annealing warm restarts.')

    weights= fields.Str(
        required=False,
        missing=None,
        description= 'path to model weights if using pretrained weights.')

    resume_training= fields.Bool(
        required=False,
        missing=False,
        enum=[True,False],
        description= 'If using pretrained weights, resume trining from the last step of the provided checkpoint.')

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
        required=True,
        description= 'Model timestamp to be used for prediction. To see the available timestamp,'
              ' please run the get_metadata function.')

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
    args=PredictArgsSchema()
    print(args)
