# 1xN Pattern for Pruning Convolutional Neural Networks ([paper](https://arxiv.org/abs/2105.14713)) ![]( https://visitor-badge.glitch.me/badge?page_id=lmbxmu.1xn).
 
Pytorch implementation of our paper accepted by TPAMI 2022 -- "1xN Pattern for Pruning Convolutional Neural Networks".


## 1) 1&times;N Block Pruning

<div align=center><img src="https://raw.githubusercontent.com/lmbxmu/1xN/master/images/comparison.jpg" height = "60%" width = "70%"/></div>


### Requirements

-  Python 3.7
-  Pytorch >= 1.0.1
-  CUDA = 10.0.0

### Code Running

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
The pre-trained models can be downloaded at [MobileNet-V1](https://drive.google.com/file/d/1bTk8nhghmiQNiE56ZdLioGbB7tNWtFMP/view?usp=sharing), [MobileNet-V2](https://drive.google.com/file/d/1rOMvEr_KCAabpAQ1PFDmRvvPB7y2Uhg8/view?usp=sharing), [MobileNet-V3-Large](https://drive.google.com/file/d/1uMiLtl9hADvpGht7d747geSu-Ssrzdll/view?usp=sharing), [MobileNet-V3-Small](https://drive.google.com/file/d/1s0kMjO2_61IgaIigJWjRzH5XPqRJyWiI/view?usp=sharing) and [ResNet-50](https://drive.google.com/file/d/1IKcmsiqj_uEtKKawtulFuktEX0cAbBcC/view?usp=sharing).
### Accuracy Performance

**Table 1: Performance comparison of our 1×N block sparsity against weight pruning and filter pruning (p = 50%).**

| MobileNet-V1   | Top-1 Acc. | Top-5 Acc. | Model Link                                                   |
| :------------- | ---------- | ---------- | ------------------------------------------------------------ |
| Weight Pruning | 70.764     | 89.592     | [Pruned Model](https://drive.google.com/drive/folders/1VDKwm8E2pfiZrIChFDBhsJtRx1LOwgR4?usp=sharing) |
| Filter Pruning | 65.348     | 86.264     | [Pruned Model](https://drive.google.com/drive/folders/19pUzgrenVMt98y496qnTV2HoIAxM8Adp?usp=sharing) |
| 1 x 2 Block    | 70.281     | 89.370     | [Pruned Model](https://drive.google.com/drive/folders/1R8qrP84-cfAZ5fH1FKrUi3o-V51gfZK7?usp=sharing) |
| 1 x 4 Block    | 70.052     | 89.056     | [Pruned Model](https://drive.google.com/drive/folders/1oWx-ceweDIjlCFF9bhujbNmjV6oFEmNg?usp=sharing) |
| 1 x 8 Block    | 69.908     | 89.027     | [Pruned Model](https://drive.google.com/drive/folders/1N_bEAW5B04ji2t3F24vqkKFI27kxW2qs?usp=sharing) |
| 1 x 16 Block   | 69.559     | 88.933     | [Pruned Model](https://drive.google.com/drive/folders/1KiCTYfasGGqhROp3SA_82tp7Q4WAr5eF?usp=sharing) |
| 1 x 32 Block   | 69.541     | 88.801     | [Pruned Model](https://drive.google.com/drive/folders/1QqXAt60Wn9n8vY7EZ4aXNnSwEnRbJt83?usp=sharing) |

| MobileNet-V2   | Top-1 Acc. | Top-5 Acc. | Model Link                                                   |
| :------------- | ---------- | ---------- | ------------------------------------------------------------ |
| Weight Pruning | 71.146     | 89.872     | [Pruned Model](https://drive.google.com/drive/folders/1FGQFjEkRoSkg3qTAXqjZraDVP4sEHFjZ?usp=sharing) |
| Filter Pruning | 66.730     | 87.190     | [Pruned Model](https://drive.google.com/drive/folders/1tKQIyc2bcdF68OlADv55JdIRTKU78VtS?usp=sharing) |
| 1 x 2 Block    | 70.233     | 89.417     | [Pruned Model](https://drive.google.com/drive/folders/1IYK4I-c334uTovdUWaE_42S0p5egoiNk?usp=sharing) |
| 1 x 4 Block    | 60.706     | 89.165     | [Pruned Model](https://drive.google.com/drive/folders/1T1gyvPwq2qkr1S-EhghxRTadT3_ZID8G?usp=sharing) |
| 1 x 8 Block    | 69.372     | 88.862     | [Pruned Model](https://drive.google.com/drive/folders/13h0VLJP73Htch4MHbQr34RULnuQjG584?usp=sharing) |
| 1 x 16 Block   | 69.352     | 88.708     | [Pruned Model](https://drive.google.com/drive/folders/15koTWszUzyINmMqtMaW0NSOdf8bhiPQr?usp=sharing) |
| 1 x 32 Block   | 68.762     | 88.425     | [Pruned Model](https://drive.google.com/drive/folders/1zcATTKj4eZUTf81DJeRslRP7jMuhQTmQ?usp=sharing) |

| MobileNet-V3-small | Top-1 Acc. | Top-5 Acc. | Model Link                                                   |
| :----------------- | ---------- | ---------- | ------------------------------------------------------------ |
| Weight Pruning     | 66.376     | 86.868     | [Pruned Model](https://drive.google.com/drive/folders/1TopFbIgopEDdoQ8qf_FE4UuaKRhq-COn?usp=sharing) |
| Filter Pruning     | 59.054     | 81.713     | [Pruned Model](https://drive.google.com/drive/folders/1fgaPjCe1pOLKvfGbm89hNeGzQ4qpTzs_?usp=sharing) |
| 1 x 2 Block        | 65.380     | 86.060     | [Pruned Model](https://drive.google.com/drive/folders/1OnTNcN5DMVAwaGSY-PO7ZFp831Z368aX?usp=sharing) |
| 1 x 4 Block        | 64.465     | 85.495     | [Pruned Model](https://drive.google.com/drive/folders/13JDlVJO5WgKJLSA5hR07U-5LJdNXWsvz?usp=sharing) |
| 1 x 8 Block        | 64.101     | 85.274     | [Pruned Model](https://drive.google.com/drive/folders/1Pi_OQNspaGcAo58hiqPkqgWHN3s8votq?usp=sharing) |
| 1 x 16 Block       | 63.126     | 84.203     | [Pruned Model](https://drive.google.com/drive/folders/1dQKHqo5NscbdWSDbymgob3yEaHsLWnbb?usp=sharing) |
| 1 x 32 Block       | 62.881     | 83.982     | [Pruned Model](https://drive.google.com/drive/folders/1izGPQphLYrRznHau951e3GGIWD_0W7xR?usp=sharing) |

| MobileNet-V3-large | Top-1 Acc. | Top-5 Acc. | Model Link                                                   |
| :----------------- | ---------- | ---------- | ------------------------------------------------------------ |
| Weight Pruning     | 72.897     | 91.093     | [Pruned Model](https://drive.google.com/drive/folders/1vn9p7G4g7fZ2IP28Cm2gxMdzHJgeKqjP?usp=sharing) |
| Filter Pruning     | 69.137     | 89.097     | [Pruned Model](https://drive.google.com/drive/folders/1zasJdeBGJY-xPJ6n9fm6cgpeUpGHFH-z?usp=sharing) |
| 1 x 2 Block        | 72.120     | 90.677     | [Pruned Model](https://drive.google.com/drive/folders/1M6PHEH60b8tjS594jFIEVwKtfyD3O9NJ?usp=sharing) |
| 1 x 4 Block        | 71.935     | 90.458     | [Pruned Model](https://drive.google.com/drive/folders/1wtCox9kqGa7f6B9Z0c-D9cKX_VmQYkUo?usp=sharing) |
| 1 x 8 Block        | 71.478     | 90.163     | [Pruned Model](https://drive.google.com/drive/folders/1pmH-Lgec5tki9OE2nut_c8q8rvC_HqAJ?usp=sharing) |
| 1 x 16 Block       | 71.112     | 90.129     | [Pruned Model](https://drive.google.com/drive/folders/1pmH-Lgec5tki9OE2nut_c8q8rvC_HqAJ?usp=sharing) |
| 1 x 32 Block       | 70.769     | 89.696     | [Pruned Model](https://drive.google.com/drive/folders/1aOvNBbjbwe1LcRBaqLacMkrmBHcd-Mbg?usp=sharing) |


<div align=center><img src="https://github.com/lmbxmu/1xN/blob/master/images/rates.jpg" height = "60%" width = "70%"/></div>

Besides, we provide the raw data for plotting the above figures in `./raw_data_fig4`. For example, run `python ./raw_data_fig4/resnet50_top1.py` to plot top-1 accuracy of ResNet-50 pruned by different methods.

More links for pruned models under different pruning rates and their training logs can be found in [MobileNet-V2](https://drive.google.com/drive/folders/1Exbxsf-VlJUmaLxxeJ0EmP0ymkpZK5mQ?usp=sharing) and [ResNet-50](https://drive.google.com/drive/folders/1TV2kX5Xh-OQfk_rIgTjEdUPzKMrlGeN-?usp=sharing).

**Table 2: Performance studies of our 1×N pruning with kernel-wise pruning.**

|  ResNet-50  | Top-1 Acc. | Top-5 Acc. | Model Link                                                   |
| :------------- | ---------- | ---------- | ------------------------------------------------------------ |
| 1x4 Block | 76.506     | 93.239     | [Pruned Model](https://drive.google.com/drive/folders/1B9foigLEX_Qtff_0ZpJYDnLw7Z79DBp9?usp=sharing) |
| kernel (random) | 74.834     | 92.178      | [Pruned Model](https://drive.google.com/drive/folders/1a8vNukBV_TKE8a_t_WF062QuidyYpFAx?usp=sharing) |
| kernel ($\ell_1$)   | 75.370 | 92.582     | [Pruned Model](https://drive.google.com/drive/folders/1W0F55TturcJBALSR_L3e_i69FY6C4lVR?usp=sharing) |

### Evaluate our models

To verify the performance of our pruned models, download our pruned models from the links provided above and run the following command:

```
python imagenet.py \
--gpus 0 \
--arch mobilenet_v1 (or mobilenet_v2 or mobilenet_v3_large or mobilenet_v3_small) \
--data_path [DATA_PATH] \
--conv_type DenseConv \
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
  --conv_type           Importance criterion of filters. Default: BlockL1Conv. optional: BlockRandomConv, DenseConv
  --pr_target           Pruning rate. default:0.5
  --full                If this parameter exists, prune fully-connected layer.
  --N                   Consecutive N kernels for removal (see paper for details).
  --rearrange           If this parameter exists, filters will be rearranged (see paper for details).
  --export_onnx         If this parameter exists, export onnx model.
```

## 2）Filter Rearrangement

<div align=center><img src="https://github.com/lmbxmu/1xN/blob/master/images/rearrangement.jpg" height = "60%" width = "70%"/></div>

**Table 2: Performance studies of our 1×N block sparsity with and without filter rearrangement (p=50%).**

| N = 2        | Top-1 Acc. | Top-5 Acc. | Model Link                                                   |
| :----------- | ---------- | ---------- | ------------------------------------------------------------ |
| w/o Rearange | 69.900     | 89.296     | [Pruned Model](https://drive.google.com/drive/folders/1U7DnIkJ9aMxRhMGQETVur21kHjIhzb-6?usp=sharing) |
| Rearrange    | 70.233     | 89.417     | [Pruned Model](https://drive.google.com/drive/folders/1qRJDeYr5QdP_qtvKdkF0-RF0La3HbNLR?usp=sharing) |

| N = 4        | Top-1 Acc. | Top-5 Acc. | Model Link                                                   |
| :----------- | ---------- | ---------- | ------------------------------------------------------------ |
| w/o Rearange | 69.521     | 88.920     | [Pruned Model](https://drive.google.com/drive/folders/1wCZKyz-PmM0dvydLqQYqEoS4Mq7to9KC?usp=sharing) |
| Rearrange    | 69.579     | 88.944     | [Pruned Model](https://drive.google.com/drive/folders/1pslyMvs_LR_SE6coLq1a_uMwg6t65aj-?usp=sharing) |

| N = 8        | Top-1 Acc. | Top-5 Acc. | Model Link                                                   |
| :----------- | ---------- | ---------- | ------------------------------------------------------------ |
| w/o Rearange | 69.206     | 88.608     | [Pruned Model](https://drive.google.com/drive/folders/1TLkaKksJXDAYBeXzGXVafYlZ_UPeSQiI?usp=sharing) |
| Rearrange    | 69.372     | 88.862     | [Pruned Model](https://drive.google.com/drive/folders/1S9UzvbN-16MezlBb9x98Id-XaWDs4cei?usp=sharing) |

| N = 16       | Top-1 Acc. | Top-5 Acc. | Model Link                                                   |
| :----------- | ---------- | ---------- | ------------------------------------------------------------ |
| w/o Rearange | 68.971     | 88.399     | [Pruned Model](https://drive.google.com/drive/folders/1l2L-VEX10Kl4QtUtyivUtGunvl5_W1se?usp=sharing) |
| Rearrange    | 69.352     | 88.708     | [Pruned Model](https://drive.google.com/drive/folders/17PiNdI9CGjtDBpEPmLDgyen764U9HP2P?usp=sharing) |

| N = 32       | Top-1 Acc. | Top-5 Acc. | Model Link                                                   |
| :----------- | ---------- | ---------- | ------------------------------------------------------------ |
| w/o Rearange | 68.431     | 88.315     | [Pruned Model](https://drive.google.com/drive/folders/1w0ERwQ6X7KL3srdiit-ls6Ppqv4NXGZt?usp=sharing) |
| Rearrange    | 68.762     | 88.425     | [Pruned Model](https://drive.google.com/drive/folders/1e8VehUWw9XU9a4qvP2lYBmoM5c_AlDnw?usp=sharing) |



## 3）Encoding and Decoding Efficiency

<div align=center><img src="https://raw.githubusercontent.com/lmbxmu/1xN/master/images/sparse.jpg" height = "60%" width = "70%"/></div>

### Performance and latency comparison

<div align=center><img src="https://github.com/lmbxmu/1xN/blob/master/images/acceleration.jpg" height = "60%" width = "70%"/></div>

Our sparse convolution implementation has been released to [TVM](https://github.com/apache/tvm) community.

To verify the performance of our pruned models, convert onnx model and run the following command:

```
python model_tune.py \
--onnx_path [ONNX_MODEL_PATH] \
--bsr 4 \
--bsc 1 \
--sparsity 0.5
```

The detail tuning setting is referred to [TVM](https://tvm.apache.org/docs/tutorials/auto_scheduler/tune_network_arm.html).


## 4）Contact

Any problem regarding this code re-implementation, please contact the first author: lmbxmu@stu.xmu.edu.cn or the second author: yuxinzhang@stu.xmu.edu.cn.

Any problem regarding the sparse convolution implementation, please contact the third author: xiamenlyc@gmail.com.


