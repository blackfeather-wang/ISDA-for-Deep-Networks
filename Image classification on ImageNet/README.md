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

## Results & Pre-trained models

|Model|Params|Baseline|ISDA|Model|
|-----|------|-----|-----|-----|
|ResNet-50  |25.6M |23.0|**21.9**|[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/2ccd502cbf774b40a226/?dl=1)|
|ResNet-101 |44.6M |21.7|**20.8**|[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/4ac40c241b8941619109/?dl=1)|
|ResNet-152 |60.3M |21.3|**20.3**|[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/7707e8709b70446fb65e/?dl=1)|
|DenseNet-BC-121 |8.0M |23.7|**23.2**|[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/e5baa6f0ac2a42ba8421/?dl=1)|
|DenseNet-BC-265 |33.3M |21.9|**21.2**|[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/ba91c2d5ce7b4650a143/?dl=1)|
|ResNeXt50, 32x4d |25.0M|22.5|**21.3**|[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/3ae2de3bdd13495ab181/?dl=1)|
|ResNeXt101, 32x8d|88.8M|21.1|**20.1**|[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/7dcca2bd9cfa426bb52d/?dl=1)|
