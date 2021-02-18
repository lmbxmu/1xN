python imagenet.py \
--arch mobilenet_v2 \
--data_path /media/disk1/ImageNet2012 \
--job_dir ./experiment/imagenet/mobv2block \
--pretrained_model ../pre-train/mobilenet_v2.pth \
--lr 0.05 \
--num_epochs 180 \
--weight_decay 4e-5 \
--gpus 0 \
--train_batch_size 256 \
--eval_batch_size 256 \
--conv_type Block16L1Conv \
--layerwise uniform

python cifar.py --arch vgg --data_path ../Datasets/Cifar --job_dir ../Experiment/ICCV21_Mingbao/vgg_1 --pretrained_model ../pre-train/vgg16_cifar10.pt --lr 0.01 --num_epochs 150 --lr 0.01 --lr_decay_step 50 100 --num_epochs 150 --weight_decay 5e-4 --conv_type Block16UnfoldConv --layerwise l1 --pr_target 0.75 --gpus 0 1 2 3

python imagenet.py --arch mobilenet_v2 --data_path /media/MEMORY_DATA --job_dir ../Experiment/ICCV21_Mingbao/test  --pretrained_model ../pre-train/mobilenet_v2.pth.tar --lr 0.05 --num_epochs 180 --weight_decay 4e-5 --gpus 0 --train_batch_size 2 --eval_batch_size 2 --conv_type Block16UnfoldConv --layerwise uniform --debug
