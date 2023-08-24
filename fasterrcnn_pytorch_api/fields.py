import json
from webargs import fields, validate
from marshmallow import Schema, ValidationError

from fasterrcnn_pytorch_api import configs


class MyCustomFieldForJson(fields.String):
    def __init__(self, *args, **kwargs):
        self.metadata = kwargs.get("metadata", {})
        self.metadata["description"] = kwargs.get("description")
        super().__init__(*args, **kwargs)

    def _deserialize(self, value, attr, data, **kwargs):
        try:
            return json.loads(value)
        except json.JSONDecodeError as err:
            raise ValidationError(f"Invalid JSON: `{err}`")

    def _validate(self, value):
        if not isinstance(value, dict):
            raise ValidationError(
                "Invalid value. Expected a dictionary."
            )

        for k1, v1 in value.items():
            if not isinstance(v1, dict):
                raise ValidationError(
                    f"Invalid value for {k1}. Expected a dictionary."
                )

            for k2, v2 in v1.items():
                if k2 == "p":
                    if not isinstance(v2, float) or not (
                        0 <= v2 <= 1.0
                    ):
                        raise ValidationError(
                            f"Invalid value for 'p' in {k2}: {v2}."
                            "It must be a float between 0 and 1."
                        )
                elif k2 in [
                    "max_w_size",
                    "max_h_size",
                    "num_holes",
                    "blur_limit",
                ]:
                    if not isinstance(v2, int) or v2 < 0:
                        raise ValidationError(
                            f"Invalid value for '{k2}' in {k1}: {v2}."
                            " It must be a non-negative integer."
                        )
                elif k2 in ["scale_limit", "shift_limit"]:
                    if not isinstance(v2, float) or not isinstance(
                        v2, (float, float)
                    ):
                        raise ValidationError(
                            f"Invalid value for '{k2}': {v2}."
                            "It must be a float or (float, float)."
                        )
                elif k2 == "rotate_limit":
                    if not isinstance(v2, int) or not isinstance(
                        v2, (int, int)
                    ):
                        raise ValidationError(
                            f"Invalid value for '{k2}': {v2}."
                            "It must be an int or (int, int)."
                        )


class TrainArgsSchema(Schema):
    class Meta:
        ordered = True

    model = fields.Str(
        enum=configs.BACKBONES,
        required=True,
        description="Name of the model.",
    )

    data_config = fields.Str(
        required=True, description="Path to the data_config.yaml"
        "file e.g.  my_dataset/data_config.yaml"
    )

    use_train_aug = fields.Bool(
        required=False,
        missing=False,
        enum=[True, False],
        description="Whether to use train augmentation, uses"
        "some advanced augmentation that may make training"
        "difficult when used with mosaic. If true, it uses"
        "the options in aug_training_option. You can change"
        "that to have custom augmentation.",
    )

    aug_training_option = MyCustomFieldForJson(
        description="Augmentation options.\n"
        "blur_limit (int) - maximum kernel size for blurring"
        "the input image.\n"
        "p (float) - probability of applying the transform.\n"
        "shift_limit ((float, float) or float) - shift factor range for"
        "both height and width.\n"
        "scale_limit ((float, float) or float) - scaling factor range.\n"
        "rotate_limit ((int, int) or int) - rotation range.\n"
        "num_holes (int) - number of regions to zero out.\n"
        "max_h_size (int) - maximum height of the hole.\n"
        "max_w_size (int) - maximum width of the hole.\n",
        missing=json.dumps(configs.DATA_AUG_OPTION),
    )

    device = fields.Bool(
        required=False,
        missing=True,
        enum=[True, False],
        description="Computation/training device, default is GPU if"
        "GPU present.",
    )

    epochs = fields.Int(
        required=False,
        missing=4,
        description="Number of epochs to train.",
    )

    workers = fields.Int(
        required=False,
        missing=4,
        description="Number of workers for data processing/transforms"
        "/augmentations.",
    )

    batch = fields.Int(
        required=False,
        missing=4,
        description="Batch size to load the data.",
    )

    lr = fields.Float(
        required=False,
        missing=0.001,
        description="Learning rate for training.",
    )

    imgsz = fields.Int(
        required=False,
        missing=640,
        description="Image size to feed to the network.",
    )

    no_mosaic = fields.Bool(
        required=False,
        missing=True,
        enum=[True, False],
        description="Pass this to not use mosaic augmentation.",
    )

    cosine_annealing = fields.Bool(
        required=False,
        missing=True,
        enum=[True, False],
        description="Use cosine annealing warm restarts.",
    )

    weights = fields.Str(
        required=False,
        missing=None,
        description="Path to model weights if using custom pretrained weights."
        "The name of the directory that contains the checkpoint within"
        'the "model" directory. To see the list of available trained models'
        'please refere to metadata methods.',
    )

    resume_training = fields.Bool(
        required=False,
        missing=False,
        enum=[True, False],
        description="If using custom pretrained weights, resume training from"
        "the last step of the provided checkpoint. If True, the path to"
        "the weights should be specified in the argument weights.",
    )

    square_training = fields.Bool(
        required=False,
        missing=True,
        enum=[True, False],
        description="Resize images to square shape instead of aspect ratio"
        "resizing for single image training. For mosaic training,"
        "this resizes single images to square shape first then puts"
        "them on a square canvas.",
    )

    disable_wandb = fields.Bool(
        required=False,
        missing=True,
        description="Whether to use WandB for logging.",
    )

    seed = fields.Int(
        required=False,
        missing=0,
        description="Global seed for training.",
    )


class PredictArgsSchema(Schema):
    class Meta:
        ordered = True

    input = fields.Field(
        required=True,
        type="file",
        location="form",
        description="Input either an image or a video.\n"
        "video must be in the format MP4, AVI, MKV, MOV, WMV, FLV, WebM.\n"
        "Images must be in the format JPEG, PNG, BMP, GIF, TIFF, PPM,"
        "EXR, WebP.",
    )

    timestamp = fields.Str(
        required=False,
        missing=None,
        description="Model timestamp to be used for prediction. To see "
        "the available timestamp, please run the get_metadata function."
        "If no timestamp is given, the model will be loaded from COCO.",
    )

    model = fields.Str(
        required=False,
        missing="fasterrcnn_resnet50_fpn_v2",
        enum=configs.BACKBONES,
        description="Please provide the name of the model you want to use"
        "for inference. If you have specified neither timestamp nor model"
        "name, the default model 'fasterrcnn_resnet50_fpn_v2' is loaded.",
    )

    threshold = fields.Float(
        required=False,
        missing=0.5,
        description="Detection threshold.",
    )

    imgsz = fields.Int(
        required=False,
        missing=640,
        description="Image size to feed to the network.",
    )

    device = fields.Bool(
        required=False,
        missing=True,
        enum=[True, False],
        description="Computation device, default is GPU if GPU is present.",
    )

    no_labels = fields.Bool(
        required=False,
        missing=False,
        enum=[True, False],
        description="Visualize output only if this argument is passed",
    )

    square_img = fields.Bool(
        required=False,
        missing=True,
        enum=[True, False],
        description="Whether to use square image resize, else use aspect ratio"
        "resize.",
    )

    accept = fields.Str(
        missing="application/json",
        location="headers",
        validate=validate.OneOf(
            ["application/json", "image/png", "video/mp4"]
        ),
        description="Returns a PNG file with detection results or a JSON with"
        "the prediction.",
    )


if __name__ == "__main__":
    pass
