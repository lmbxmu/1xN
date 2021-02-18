# 1-N


## Running Code

### Requirements

-  Pytorch >= 1.0.1
-  CUDA = 10.0.0

### Pre-train Models

Additionally, we provide several pre-trained models used in our experiments.

#### CIFAR-10

| [VGG16](https://drive.google.com/open?id=1pz-_0CCdL-1psIQ545uJ3xT6S_AAnqet) | [ResNet56](https://drive.google.com/open?id=1pt-LgK3kI_4ViXIQWuOP0qmmQa3p2qW5) | [ResNet110](https://drive.google.com/open?id=1Uqg8_J-q2hcsmYTAlRtknCSrkXDqYDMD) |[GoogLeNet](https://drive.google.com/open?id=1YNno621EuTQTVY2cElf8YEue9J4W5BEd) |

#### ImageNet

 | [ResNet50](https://download.pytorch.org/models/resnet50-19c8e357.pth) |[MobileNet-V1](https://hanlab.mit.edu/projects/amc/external/mobilenet_imagenet.pth.tar) | 

### 

```shell
python imagenet.py --arch mobilenet_v2 --data_path /media/MEMORY_DATA --job_dir ../Experiment/test  --pretrained_model ../pre-train/mobilenet_v2.pth.tar --lr 0.05 --num_epochs 180 --weight_decay 4e-5 --gpus 0 --train_batch_size 256 --eval_batch_size 256 --conv_type Block16UnfoldConv --layerwise uniform --debug
```


### Arguments

```shell
optional arguments:
  -h, --help            show this help message and exit
  --gpus                Select gpu_id to use. default:[0]
  --data_set            Select dataset to train. default:cifar10
  --data_path           The dictionary where the input is stored.
                        default:/home/lishaojie/data/cifar10/
  --job_dir             The directory where the summaries will be stored.
                        default:./experiments
  --reset               Reset the directory?
  --resume              Load the model from the specified checkpoint.
  --refine              Path to the model to be fine tuned.
  --arch                Architecture of model. For cifar: vgg_cifar, resnet_cifar, googlenet, densenet .For ImageNet :vgg,resnet,googlenet,densenet
  --cfg                 Detail architecuture of model. For cifar: vgg16, resnet56/110, googlenet. For ImageNet :vgg16,resnet18/34/50/101/152
  --num_epochs          The num of epochs to train. default:150
  --train_batch_size    Batch size for training. default:128
  --eval_batch_size     Batch size for validation. default:100
  --momentum            Momentum for MomentumOptimizer. default:0.9
  --lr LR               Learning rate for train. default:1e-2
  --lr_decay_step       The iterval of learn rate decay for cifar. default:100 150
  --lr_decay_freq       The frequecy of learn rate decay for Imagenet. default:30
  --weight_decay        The weight decay of loss. default:5e-4
  --lr_type             lr scheduler. default: step. optional:exp/cos/step/fixed
  --use_dali            If this parameter exists, use dali module to load ImageNet data.
  --conv_type           Conv type of conv layer. Default: DenseConv. optional: Block16L1Conv/Block16RandomConv/Block16UnfoldConv

```

## Tips

Any problem, free to contact yxzhangxmu@163.com.