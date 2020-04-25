# Image classification on ImageNet

## Requirements
- python 3.7
- torch 1.0.1
- torchvision 0.2.2


## Run

Train ResNet-50 on ImageNet

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python imagenet_DDP.py PATH_TO_DATASET --model resnet50 --batch-size 512 --lr 0.2 --epochs 300 --lambda_0 7.5 --workers 32 --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0
```

Train DenseNet-121 on ImageNet

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python imagenet_DDP.py PATH_TO_DATASET --model densenet121 --batch-size 512 --lr 0.2 --epochs 300 --lambda_0 1 --workers 32 --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0
```

Evaluate pre-trained model on ImageNet

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python imagenet_DDP.py PATH_TO_DATASET --model MODEL_NAME --resume PATH_TO_MODEL -e --batch-size 512 --lr 0.2 --epochs 300 --lambda_0 0 --workers 32 --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0
```
