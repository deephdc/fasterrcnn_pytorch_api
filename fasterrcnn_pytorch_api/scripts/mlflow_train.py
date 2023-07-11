"""
USAGE

# training with Faster RCNN ResNet50 FPN model without mosaic or any other augmentation:
python train.py --model fasterrcnn_resnet50_fpn --epochs 2 --data data_configs/voc.yaml --no-mosaic --batch 4

# Training on ResNet50 FPN with custom project folder name with mosaic augmentation (ON by default):
python train.py --model fasterrcnn_resnet50_fpn --epochs 2 --data data_configs/voc.yaml --name resnet50fpn_voc --batch 4

# Training on ResNet50 FPN with custom project folder name with mosaic augmentation (ON by default) and added training augmentations:
python train.py --model fasterrcnn_resnet50_fpn --epochs 2 --use-train-aug --data data_configs/voc.yaml --name resnet50fpn_voc --batch 4
dist-url: is not used in the main function and it is not needed
"""
import argparse
import tempfile
from threading import main_thread

from fasterrcnn_pytorch_training_pipeline.torch_utils.engine import (
    train_one_epoch, evaluate, utils
)

from fasterrcnn_pytorch_training_pipeline.datasets import (
    create_train_dataset, create_valid_dataset, 
    create_train_loader, create_valid_loader
)
from fasterrcnn_pytorch_training_pipeline.models.create_fasterrcnn_model import create_model
from fasterrcnn_pytorch_training_pipeline.utils.general import (
    Averager, 
    save_model ,
    show_tranformed_image,
      save_model_state, SaveBestModel,
    yaml_save, init_seeds
)


 
from  torch.utils.data import (
     RandomSampler, SequentialSampler
)
import torch
import yaml
import numpy as np
import torchinfo
import os

#from prettytable import PrettyTable

import matplotlib.pyplot as plt
from scipy import stats

import mlflow
import mlflow.pytorch

import getpass

from datetime import datetime


from mlflow.tracking import MlflowClient
from mlflow import log_metric, log_param, log_artifacts


# Remote MLFlow server
MLFLOW_REMOTE_SERVER="http://mlflow.dev.ai4eosc.eu"
#Set the MLflow server and backend and artifact stores
mlflow.set_tracking_uri(MLFLOW_REMOTE_SERVER)

# for direct API calls via HTTP we need to inject credentials
MLFLOW_TRACKING_USERNAME = 'mlflow_user'
MLFLOW_TRACKING_PASSWORD =  getpass.getpass()  # inject password by typing manually
USERNAME = "lisana.b" # User who is logging the experiment, if not set then the default value of a user will be your local username
#set the environmental vars to allow 'mlflow_user' to track experiments using MLFlow

# for MLFLow-way we have to set the following environment variables
os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD

os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
os.environ["LOGNAME"] = USERNAME  
#nginx credentials

run_name = datetime.now().strftime("%Y%m%d_%H%M%S")


torch.multiprocessing.set_sharing_strategy('file_system')

RANK = int(os.getenv('RANK', -1))

# For same annotation colors each time.
np.random.seed(42)
def parse_opt():
    # Construct the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model', 
        default='fasterrcnn_resnet50_fpn_v2',
        help='name of the model'
    )
    parser.add_argument(
        '--data', 
        default=None,
        help='path to the data config file'
    )
    parser.add_argument(
        '-d', '--device', 
        default='cuda',
        help='computation/training device, default is GPU if GPU present'
    )
    parser.add_argument(
        '-e', '--epochs', 
        default=5,
        type=int,
        help='number of epochs to train for'
    )
    parser.add_argument(
        '-j', '--workers', 
        default=4,
        type=int,
        help='number of workers for data processing/transforms/augmentations'
    )
    parser.add_argument(
        '-b', '--batch', 
        default=4, 
        type=int, 
        help='batch size to load the data'
    )
    parser.add_argument(
        '--lr', 
        default=0.001,
        help='learning rate for the optimizer',
        type=float
    )
    parser.add_argument(
        '-ims', '--imgsz',
        default=640, 
        type=int, 
        help='image size to feed to the network'
    )
    parser.add_argument(
        '-n', '--name', 
        default=None, 
        type=str, 
        help='training result dir name in outputs/training/, (default res_#)'
    )
    parser.add_argument(
        '-vt', '--vis-transformed', 
        dest='vis_transformed', 
        action='store_true',
        help='visualize transformed images fed to the network'
    )
    parser.add_argument(
        '-nm', '--no-mosaic', 
        dest='no_mosaic', 
        action='store_true',
        help='pass this to not to use mosaic augmentation'
    )
    parser.add_argument(
        '-uta', '--use-train-aug', 
        dest='use_train_aug', 
        action='store_true',
        help='whether to use train augmentation, uses some advanced \
            augmentation that may make training difficult when used \
            with mosaic'
    )
    parser.add_argument(
        '-ca', '--cosine-annealing', 
        dest='cosine_annealing', 
        action='store_true',
        help='use cosine annealing warm restarts'
    )
    parser.add_argument(
        '-w', '--weights', 
        default=None, 
        type=str,
        help='path to model weights if using pretrained weights'
    )
    parser.add_argument(
        '-r', '--resume-training', 
        dest='resume_training', 
        action='store_true',
        help='whether to resume training, if true, \
            loads previous training plots and epochs \
            and also loads the otpimizer state dictionary'
    )
    parser.add_argument(
        '-st', '--square-training',
        dest='square_training',
        action='store_true',
        help='Resize images to square shape instead of aspect ratio resizing \
              for single image training. For mosaic training, this resizes \
              single images to square shape first then puts them on a \
              square canvas.'
    )
    parser.add_argument(
        '--world-size', 
        default=1, 
        type=int, 
        help='number of distributed processes'
    )
    parser.add_argument(
        '--dist-url',
        default='env://',
        type=str,
        help='url ysed to set up the distributed training'
    )

    parser.add_argument(
        '-dw', '--disable-wandb',
        dest="disable_wandb",
        action='store_true',
        help='whether to use the wandb'
    )
    parser.add_argument(
        '--seed',
        default=0,
        type=int ,
        help='golabl seed for training'
    )

    args = vars(parser.parse_args())
    return args

def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    
    #+print as an output in the console
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))


def main(args):
    # Initialize distributed mode.
    utils.init_distributed_mode(args)
 
    
    # Load the data configurations
    with open(args['data_config']) as file:
        data_configs = yaml.safe_load(file)

    init_seeds(args['seed'] + 1 + RANK, deterministic=True)
    
    # Settings/parameters/constants.
    TRAIN_DIR_IMAGES = os.path.normpath(data_configs['TRAIN_DIR_IMAGES'])
    TRAIN_DIR_LABELS = os.path.normpath(data_configs['TRAIN_DIR_LABELS'])
    VALID_DIR_IMAGES = os.path.normpath(data_configs['VALID_DIR_IMAGES'])
    VALID_DIR_LABELS = os.path.normpath(data_configs['VALID_DIR_LABELS'])
    CLASSES = data_configs['CLASSES']
    NUM_CLASSES = data_configs['NC']
    NUM_WORKERS = args['workers']
    if  args['device'] and torch.cuda.is_available():
         DEVICE  = torch.device('cuda:0')
    else:
         DEVICE  = torch.device('cpu')
    print("device",DEVICE)
    NUM_EPOCHS = args['epochs']
    BATCH_SIZE = args['batch']
    OUT_DIR = args['name'] 
    COLORS = np.random.uniform(0, 1, size=(len(CLASSES), 3))
    yaml_save(file_path=os.path.join(OUT_DIR, 'opt.yaml'), data=args)

    # Model configurations
    IMAGE_SIZE = args['imgsz']
    
    train_dataset = create_train_dataset(
        TRAIN_DIR_IMAGES, 
        TRAIN_DIR_LABELS,
        IMAGE_SIZE, 
        CLASSES,
        use_train_aug=args['use_train_aug'],
        no_mosaic=args['no_mosaic'],
        square_training=args['square_training']
    )
    valid_dataset = create_valid_dataset(
        VALID_DIR_IMAGES, 
        VALID_DIR_LABELS, 
        IMAGE_SIZE, 
        CLASSES,
        square_training=args['square_training']
    )
    print('Creating data loaders')
 
    train_sampler = RandomSampler(train_dataset)
    valid_sampler = SequentialSampler(valid_dataset)

    train_loader = create_train_loader(
        train_dataset, BATCH_SIZE, NUM_WORKERS, batch_sampler=train_sampler
    )
    valid_loader = create_valid_loader(
        valid_dataset, BATCH_SIZE, NUM_WORKERS, batch_sampler=valid_sampler
    )
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")
     
    # Initialize the Averager class.
    train_loss_hist = Averager()
    # Train and validation loss lists to store loss values of all
    # iterations till ena and plot graphs for all iterations.
    train_loss_list = []
    train_loss_list_epoch = []
    val_map_05 = []
    val_map = []
    start_epochs = 0

    if args['weights'] is None:
        print('Building model from scratch...')
        build_model = create_model[args['model']]
        model = build_model(num_classes=NUM_CLASSES, pretrained=True)

    # Load pretrained weights if path is provided.
    else:
        print('Loading pretrained weights...')
        
        # Load the pretrained checkpoint.
        checkpoint = torch.load(args['weights'], map_location=DEVICE) 
        keys = list(checkpoint['model_state_dict'].keys())
        ckpt_state_dict = checkpoint['model_state_dict']
        # Get the number of classes from the loaded checkpoint.
        old_classes = ckpt_state_dict['roi_heads.box_predictor.cls_score.weight'].shape[0]

        # Build the new model with number of classes same as checkpoint.
        build_model = create_model[args['model']]
        model = build_model(num_classes=old_classes)
        # Load weights.
        model.load_state_dict(ckpt_state_dict)

        # Change output features for class predictor and box predictor
        # according to current dataset classes.
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor.cls_score = torch.nn.Linear(
            in_features=in_features, out_features=NUM_CLASSES, bias=True
        )
        model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(
            in_features=in_features, out_features=NUM_CLASSES*4, bias=True
        )
        test_feat = in_features
        #FIXME: should be loaded from the last chaeckpoint or from a timestamp
        if args['resume_training']:
            print('RESUMING TRAINING FROM LAST CHECKPOINT...')
            # Update the starting epochs, the batch-wise loss list, 
            # and the epoch-wise loss list.

            if checkpoint['epoch']:
                start_epochs = checkpoint['epoch']
                print(f"Resuming from epoch {start_epochs}...")
                NUM_EPOCHS=start_epochs+NUM_EPOCHS
            if checkpoint['train_loss_list']:
                print('Loading previous batch wise loss list...')
                train_loss_list = checkpoint['train_loss_list']
            if checkpoint['train_loss_list_epoch']:
                print('Loading previous epoch wise loss list...')
                train_loss_list_epoch = checkpoint['train_loss_list_epoch']
            if checkpoint['val_map']:
                print('Loading previous mAP list')
                val_map = checkpoint['val_map']
            if checkpoint['val_map_05']:
                val_map_05 = checkpoint['val_map_05']

    model = model.to(DEVICE)
    if args['distributed']:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args['gpu']]
        )
    try:
        torchinfo.summary(
            model, device=DEVICE, input_size=(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
        )
    except:
        print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    # Get the model parameters.
    params = [p for p in model.parameters() if p.requires_grad]
    # Define the optimizer.
    optimizer = torch.optim.SGD(params, lr=args['lr'], momentum=0.9, nesterov=True)
    # optimizer = torch.optim.AdamW(params, lr=0.0001, weight_decay=0.0005)
    if args['resume_training']: 
        # LOAD THE OPTIMIZER STATE DICTIONARY FROM THE CHECKPOINT.
        print('Loading optimizer state dictionary...')
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if args['cosine_annealing']:
        # LR will be zero as we approach `steps` number of epochs each time.
        # If `steps = 5`, LR will slowly reduce to zero every 5 epochs.
        steps = NUM_EPOCHS + 10
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=steps,
            T_mult=1,
            verbose=False
        )
    else:
        scheduler = None
   
    save_best_model = SaveBestModel()
    best_valid_map=float(0)
    with mlflow.start_run(run_name = run_name) as mlflow_run: 
       #mlflow.pytorch.autolog(log_every_n_epoch=1, log_models=True, disable_for_unsupported_versions=False,  registered_model_name='submarin_animal_detection')

    #Log all the params
        mlflow.log_params(args)
        mlflow.set_tag("model",args['model'])

        
        for epoch in range(start_epochs, NUM_EPOCHS):
            
            
            train_loss_hist.reset()

            _, batch_loss_list, \
                batch_loss_cls_list, \
                batch_loss_box_reg_list, \
                batch_loss_objectness_list, \
                batch_loss_rpn_list = train_one_epoch(
                model, 
                optimizer, 
                train_loader, 
                DEVICE, 
                epoch, 
                train_loss_hist,
                print_freq=100,
                scheduler=scheduler
            )

            stats, _ = evaluate(
                model, 
                valid_loader, 
                device=DEVICE,
                save_valid_preds=False,
                out_dir=OUT_DIR,
                classes=CLASSES,
                colors=COLORS
            )

            train_loss_list.extend(batch_loss_list)
            train_loss_list_epoch.append(train_loss_hist.value)
            val_map_05.append(stats[1])
            val_map.append(stats[0])
            if val_map[-1] >  best_valid_map:
                best_valid_map = val_map[-1]  
                print("\nThe model is logged at:\n%s" % os.path.join(mlflow.get_artifact_uri(), "pytorch-model"))
                mlflow.pytorch.log_model(model , "best_model")
                      
            mlflow.pytorch.log_model(model , "checkpoint")
            model_state={
                'epoch': epoch+1,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss_list': train_loss_list,
                'train_loss_list_epoch': train_loss_list_epoch,
                'val_map': val_map,
                'val_map_05': val_map_05,
                'data':  data_configs,
                'model_name': args['model']
                }
            #print("\nThe model is logged at:\n%s" % os.path.join(mlflow.log_artifact(), "pytorch-model"))
            mlflow.pytorch.log_state_dict(model_state,  artifact_path="model_state")
                 
            
            #Log the metrics
            mlflow.log_metric('loss_cls',np.mean(np.array(batch_loss_cls_list)))
            mlflow.log_metric('loss_box_reg', np.mean(np.array(batch_loss_box_reg_list)))
            mlflow.log_metric('loss_objectness',np.mean(np.array(batch_loss_objectness_list)))
            mlflow.log_metric('loss_rpn', np.mean(np.array(batch_loss_rpn_list)))
            mlflow.log_metric('train_loss_epoch', train_loss_hist.value)
            mlflow.log_metric('val_mAP_05',  stats[1])
            mlflow.log_metric('val_mAP', stats[0])    
            
            run_id  = mlflow_run.info.run_id

            # fetch the auto logged parameters and metrics
            #print_auto_logged_info(mlflow.get_run(run_id=run_id))
            
    return    run_id
    print_auto_logged_info(run_id)
    mlflow.end_run()


if __name__ == '__main__':
    args={'model': 'fasterrcnn_squeezenet1_1',
           'data_config': '/home/ubuntu/data/fasterrcnn/fasterrcnn_pytorch_api/data/submarine_det/brackish.yaml',
           'name':'/home/ubuntu/data/fasterrcnn/fasterrcnn_pytorch_api/data/',
           'use_train_aug': False,
           'device': False,
           'epochs': 1,
           'workers': 4,
           'batch': 1,
           'lr': 0.001,
           'imgsz': 640,
           'no_mosaic': True,
           'cosine_annealing': False,
           'weights': None,
           'resume_training':False,
           'square_training': False,
           'seed':0
           }
    main(args)
    