# Visualizing deep features

## Run

(1) Prepare ImageNet dataset in PATH_TO_IMAGENET.

(2) Download the Pytorch pre-trained resnet50 models [resnet50-19c8e357.pth](https://download.pytorch.org/models/resnet50-19c8e357.pth).

(3) Estimate the covariance matrices for ISDA:

```
CUDA_VISIBLE_DEVICES=0 python cov_estimate.py \
    --data_url PATH_TO_IMAGENET \
    --train_url './resnet50/' \
    --workers 4 \
    --epochs 1 \
    --batch_size 16 \
    --print_freq 10
```

(4) Follow [https://github.com/huggingface/pytorch-pretrained-BigGAN](https://github.com/huggingface/pytorch-pretrained-BigGAN) to prepare biggan-related codes and model configurations (we use BigGAN-deep-51).

(5) Visualize deep features with the following command (replace n03028079_7422.JPEG with your own image):

```
CUDA_VISIBLE_DEVICES=0 python aug_biggan512_imagenet.py \
    --train_url './' \
    --img_dir './n03028079_7422.JPEG' \
    --aug_num 1 \
    --aug_alpha 0.2 \
    --epoch1 12000 \
    --epoch2 8000 \
    --schedule1 3000 6000 9000 \
    --schedule2 4000 6000 \
    --lr1 100 \
    --lr2 0.1 \
    --eta 5e-3 \
    --truncation 1
```
