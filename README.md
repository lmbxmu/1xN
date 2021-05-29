# 1×N Block Pattern for Network Sparsity

A pytorch re-implementation of 1×N Block Pattern for Network Sparsity.

### Requirements

-  Python 3.7
-  Pytorch >= 1.0.1
-  CUDA = 10.0.0

### Running code

To reproduce our experiments, please use the following command:

```
python imagenet.py \
--gpus 0 \
--arch mobilenet_v1 (or mobilenet_v2 or mobilenet_v3_large or mobilenet_v3_small) \
--job_dir ./experiment/ \
--data_path [DATA_PATH] \
--pretrained_model [PRETRAIN_MODEL_PATH] \
--pr_target 0.5 \
--N 4 (or 2, 8, 16, 32) \
--conv_type BlockL1Conv \
--train_batch_size 256 \
--eval_batch_size 256 \
--rearrange \
```

### Evaluate our pruned models

**Table 1: Performance studies of our 1×N block sparsity with and without filter rearrangement.**

|              | N=2                                                          | N=4                                                          | N=8                                                          | N=16                                                         | N=32                                                         |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| w/o Rearange | [Pruned Model](https://drive.google.com/drive/folders/1U7DnIkJ9aMxRhMGQETVur21kHjIhzb-6?usp=sharing) | [Pruned Model](https://drive.google.com/drive/folders/1wCZKyz-PmM0dvydLqQYqEoS4Mq7to9KC?usp=sharing) | [Pruned Model](https://drive.google.com/drive/folders/1TLkaKksJXDAYBeXzGXVafYlZ_UPeSQiI?usp=sharing) | [Pruned Model](https://drive.google.com/drive/folders/1l2L-VEX10Kl4QtUtyivUtGunvl5_W1se?usp=sharing) | [Pruned Model](https://drive.google.com/drive/folders/1w0ERwQ6X7KL3srdiit-ls6Ppqv4NXGZt?usp=sharing) |
| Rearrange    | [Pruned Model](https://drive.google.com/drive/folders/1qRJDeYr5QdP_qtvKdkF0-RF0La3HbNLR?usp=sharing) | [Pruned Model](https://drive.google.com/drive/folders/1pslyMvs_LR_SE6coLq1a_uMwg6t65aj-?usp=sharing) | [Pruned Model](https://drive.google.com/drive/folders/1S9UzvbN-16MezlBb9x98Id-XaWDs4cei?usp=sharing) | [Pruned Model](https://drive.google.com/drive/folders/17PiNdI9CGjtDBpEPmLDgyen764U9HP2P?usp=sharing) | [Pruned Model](https://drive.google.com/drive/folders/1e8VehUWw9XU9a4qvP2lYBmoM5c_AlDnw?usp=sharing) |

**Table 2:  Performance comparison of our 1×N block sparsity against weight pruning and filter pruning**

|                | MobileNet-V1                                                 | MobileNet-V2                                                 | MobileNet-V3-small                                           | MobileNet-V3-large                                           |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Weight Pruning | [Pruned Model](https://drive.google.com/drive/folders/1VDKwm8E2pfiZrIChFDBhsJtRx1LOwgR4?usp=sharing) | [Pruned Model](https://drive.google.com/drive/folders/1FGQFjEkRoSkg3qTAXqjZraDVP4sEHFjZ?usp=sharing) | [Pruned Model](https://drive.google.com/drive/folders/1TopFbIgopEDdoQ8qf_FE4UuaKRhq-COn?usp=sharing) | [Pruned Model](https://drive.google.com/drive/folders/1vn9p7G4g7fZ2IP28Cm2gxMdzHJgeKqjP?usp=sharing) |
| Filter Pruning | [Pruned Model](https://drive.google.com/drive/folders/19pUzgrenVMt98y496qnTV2HoIAxM8Adp?usp=sharing) | [Pruned Model](https://drive.google.com/drive/folders/1tKQIyc2bcdF68OlADv55JdIRTKU78VtS?usp=sharing) | [Pruned Model](https://drive.google.com/drive/folders/1fgaPjCe1pOLKvfGbm89hNeGzQ4qpTzs_?usp=sharing) | [Pruned Model](https://drive.google.com/drive/folders/1zasJdeBGJY-xPJ6n9fm6cgpeUpGHFH-z?usp=sharing) |
| 1 x 2 Block    | [Pruned Model](https://drive.google.com/drive/folders/1R8qrP84-cfAZ5fH1FKrUi3o-V51gfZK7?usp=sharing) | [Pruned Model](https://drive.google.com/drive/folders/1IYK4I-c334uTovdUWaE_42S0p5egoiNk?usp=sharing) | [Pruned Model](https://drive.google.com/drive/folders/1OnTNcN5DMVAwaGSY-PO7ZFp831Z368aX?usp=sharing) | [Pruned Model](https://drive.google.com/drive/folders/1M6PHEH60b8tjS594jFIEVwKtfyD3O9NJ?usp=sharing) |
| 1 x 4 Block    | [Pruned Model](https://drive.google.com/drive/folders/1oWx-ceweDIjlCFF9bhujbNmjV6oFEmNg?usp=sharing) | [Pruned Model](https://drive.google.com/drive/folders/1T1gyvPwq2qkr1S-EhghxRTadT3_ZID8G?usp=sharing) | [Pruned Model](https://drive.google.com/drive/folders/13JDlVJO5WgKJLSA5hR07U-5LJdNXWsvz?usp=sharing) | [Pruned Model](https://drive.google.com/drive/folders/1wtCox9kqGa7f6B9Z0c-D9cKX_VmQYkUo?usp=sharing) |
| 1 x 8 Block    | [Pruned Model](https://drive.google.com/drive/folders/1N_bEAW5B04ji2t3F24vqkKFI27kxW2qs?usp=sharing) | [Pruned Model](https://drive.google.com/drive/folders/13h0VLJP73Htch4MHbQr34RULnuQjG584?usp=sharing) | [Pruned Model](https://drive.google.com/drive/folders/1Pi_OQNspaGcAo58hiqPkqgWHN3s8votq?usp=sharing) | [Pruned Model](https://drive.google.com/drive/folders/1pmH-Lgec5tki9OE2nut_c8q8rvC_HqAJ?usp=sharing) |
| 1 x 16 Block   | [Pruned Model](https://drive.google.com/drive/folders/1KiCTYfasGGqhROp3SA_82tp7Q4WAr5eF?usp=sharing) | [Pruned Model](https://drive.google.com/drive/folders/15koTWszUzyINmMqtMaW0NSOdf8bhiPQr?usp=sharing) | [Pruned Model](https://drive.google.com/drive/folders/1dQKHqo5NscbdWSDbymgob3yEaHsLWnbb?usp=sharing) | [Pruned Model](https://drive.google.com/drive/folders/1pmH-Lgec5tki9OE2nut_c8q8rvC_HqAJ?usp=sharing) |
| 1 x 32 Block   | [Pruned Model](https://drive.google.com/drive/folders/1QqXAt60Wn9n8vY7EZ4aXNnSwEnRbJt83?usp=sharing) | [Pruned Model](https://drive.google.com/drive/folders/1zcATTKj4eZUTf81DJeRslRP7jMuhQTmQ?usp=sharing) | [Pruned Model](https://drive.google.com/drive/folders/1izGPQphLYrRznHau951e3GGIWD_0W7xR?usp=sharing) | [Pruned Model](https://drive.google.com/drive/folders/1aOvNBbjbwe1LcRBaqLacMkrmBHcd-Mbg?usp=sharing) |

To verify the performance of our pruned models, download our pruned models from the links provided above and run the following command:

```
python imagenet.py \
--gpus 0 \
--arch mobilenet_v1 (or mobilenet_v2 or mobilenet_v3_large or mobilenet_v3_small) \
--job_dir ./experiment/ \
--data_path [DATA_PATH] \
--evaluate [PRUNED_MODEL_PATH] \
--eval_batch_size 256 \
```


### Arguments

```shell
optional arguments:
  -h, --help            show this help message and exit
  --gpus                Select gpu_id to use. default:[0]
  --data_path           The dictionary where the data is stored.
  --job_dir             The directory where the summaries will be stored.
  --resume              Load the model from the specified checkpoint.
  --pretrain_model      Path of the pre-trained model.
  --pruned_model        Path of the pruned model to evaluate.
  --arch                Architecture of model. For ImageNet :mobilenet_v1, mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
  --num_epochs          The num of epochs to train. default:180
  --train_batch_size    Batch size for training. default:256
  --eval_batch_size     Batch size for validation. default:100
  --momentum            Momentum for Momentum Optimizer. default:0.9
  --lr LR               Learning rate. default:1e-2
  --lr_decay_step       The iterval of learn rate decay for cifar. default:100 150
  --lr_decay_freq       The frequecy of learn rate decay for Imagenet. default:30
  --weight_decay        The weight decay of loss. default:4e-5
  --lr_type             lr scheduler. default: cos. optional:exp/cos/step/fixed
  --use_dali            If this parameter exists, use dali module to load ImageNet data (benefit in training acceleration).
  --conv_type           Importance criterion of filters. Default: BlockL1Conv. optional: BlockRandomConv
  --pr_target           Pruning rate. default:0.5
  --full                If this parameter exists, prune fully-connected layer.
  --N                   Consecutive N kernels for removal (see paper for details).
  --rearrange           If this parameter exists, filters will be rearranged (see paper for details).
```

### 
