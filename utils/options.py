import argparse
import ast
import os
import sys
import yaml


parser = argparse.ArgumentParser(description='1xN Block Pruning')

parser.add_argument(
    "--config", 
    help="Config file to use (see configs dir)", 
    default=None)

parser.add_argument(
    '--use_dali',
    action='store_true',
    help='whether use dali module to load data'
    )

parser.add_argument(
    "--label-smoothing",
    type=float,
    help="Label smoothing to use, default 0.0",
    default=0.0
)

parser.add_argument(
    "--warmup_length", 
    default=5, 
    type=int, 
    help="Number of warmup iterations"
)

parser.add_argument(
	'--gpus',
	type=int,
	nargs='+',
	default=0,
	help='Select gpu_id to use. default:[0]',
)

parser.add_argument(
	'--pretrained_model',
	type=str,
	default=None,
	help='Path of the pre-trained model',
)

parser.add_argument(
	'--pruned_model',
	type=str,
	default=None,
	help='Path of the pruned model to evaluate',
)

parser.add_argument(
	'--data_set',
	type=str,
	default='cifar10',
	help='Select dataset to train. default:cifar10',
)

parser.add_argument(
	'--data_path',
	type=str,
	default='/home/userhome/Datasets/Cifar',
	help='The dictionary where the input is stored. default:',
)

parser.add_argument(
    '--job_dir',
    type=str,
    default='experiments/',
    help='The directory where the summaries will be stored. default:./experiments'
)

parser.add_argument(
    '--resume',
    action='store_true',
    help='Load the model from the specified checkpoint.'
)

parser.add_argument(
    '--lr_decay_step',
    type=int,
    nargs='+',
    default=[50, 100],
    help='the iterval of learn rate. default:50, 100'
)

## Training
parser.add_argument(
    '--arch',
    type=str,
    default='mobilenet_v2',
    help='Architecture of model. default:mobilenet_v2. optional:mobilenet_v1, mobilenet_v3_small, mobilenet_v3_large.'
)

parser.add_argument(
    '--num_epochs',
    type=int,
    default=180,
    help='The num of epochs to train. default:180'
)

parser.add_argument(
    '--train_batch_size',
    type=int,
    default=256,
    help='Batch size for training. default:256'
)

parser.add_argument(
    '--eval_batch_size',
    type=int,
    default=256,
    help='Batch size for validation. default:256'
)

parser.add_argument(
    '--momentum',
    type=float,
    default=0.9,
    help='Momentum for MomentumOptimizer. default:0.9'
)

parser.add_argument(
    '--lr',
    type=float,
    default=0.01,
    help='Learning rate for train. default:0.1'
)

parser.add_argument(
    "--optimizer", 
    help="Which optimizer to use", 
    default="sgd"
)

parser.add_argument(
    "--lr_policy", 
    default="cos", 
    help="Policy for the learning rate."
)

parser.add_argument(
    "--lr_adjust", 
    default=30, type=int, 
    help="Interval to drop lr"
)
parser.add_argument(
    "--lr_gamma", 
    default=0.1, 
    type=float, 
    help="Multistep multiplier"
)

parser.add_argument(
    '--weight_decay',
    type=float,
    default=4e-5,
    help='The weight decay of loss. default:1e-4'
)

parser.add_argument(
    '--random_rule',
    type=str,
    default='random_pretrain',
    help='Weight initialization criterion after random clipping. default:default optional:default,random_pretrain,l1_pretrain'
)

parser.add_argument(
    '--test_only',
    action='store_true',
    help='Test only?'
)

parser.add_argument(
    '--rearrange',
    action='store_true',
    help='Rearrange the 1*N block using same-size kmeans.'
)

parser.add_argument(
    "--nesterov",
    default=False,
    action="store_true",
    help="Whether or not to use nesterov for SGD",
)

parser.add_argument(
    "--first-layer-type", 
    type=str, 
    default=None, 
    
    help="Conv type of first layer"
)

parser.add_argument(
    "--conv_type", 
    type=str, 
    default='BlockL1Conv', 
    help="Conv type of conv layer. Default: BlockL1Conv. optional: BlockRandomConv"
)

parser.add_argument(
    "--bn_type", 
    default='LearnedBatchNorm', 
    help="BatchNorm type"
)

parser.add_argument(
    "--init", 
    default="kaiming_normal", 
    help="Weight initialization modifications"
)

parser.add_argument(
    "--mode", 
    default="fan_in", 
    help="Weight initialization mode"
)

parser.add_argument(
    "--nonlinearity", 
    default="relu", 
    help="Nonlinearity used by initialization"
)

parser.add_argument(
    '--pr_target',
    type=float,
    default=0.5,
    help='Target_pruning rate of parameters. default:0.5'
)

parser.add_argument(
    '--debug',
    action='store_true',
    help='input to open debug state'
)

parser.add_argument(
    "--full", 
    action="store_true", 
    help="prune full-connect layer"
)

parser.add_argument(
    "--N", 
    default=16, 
    type=int, 
    help="kernel number of the block 1 x N pattern"
)

parser.add_argument(
    "--export_onnx", 
    action="store_true", 
    help="export onnx model"
)

args = parser.parse_args()

