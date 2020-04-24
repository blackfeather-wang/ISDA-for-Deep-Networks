# Implicit Semantic Data Augmentation for Deep Networks **(NeurIPS 2019)**

Code for the paper [Implicit Semantic Data Augmentation for Deep Networks](https://arxiv.org/abs/1909.12220)

Please go to the folder [Image classification on CIFAR](https://github.com/blackfeather-wang/ISDA-for-Deep-Networks/tree/master/Image%20classification%20on%20CIFAR), [Image classification on ImageNet](https://github.com/blackfeather-wang/ISDA-for-Deep-Networks/tree/master/Image%20classification%20on%20ImageNet) and [Semantic segmentation on Cityscapes](https://github.com/blackfeather-wang/ISDA-for-Deep-Networks/tree/master/Semantic%20segmentation%20on%20Cityscapes) for specific docs.

**Update on 2020/04/24: Release Code for Image Classification on ImageNet and Semantic Segmentation on Cityscapes.**

## Introduction

In this paper, we propose a novel implicit semantic data augmentation (ISDA) approach to complement traditional augmentation techniques like flipping, translation or rotation.
ISDA consistently improves the generalization performance of popular deep networks on supervised & semi-supervised image classification, semantic segmentation, object detection and instance segmentation.

<p align="center">
    <img src="ISDA-overview.png" height="309" width= "900">
</p>


## Citation

If you find this work useful or use our codes in your own research, please use the following bibtex:

```
@inproceedings{NIPS2019_9426,
        title = {Implicit Semantic Data Augmentation for Deep Networks},
       author = {Wang, Yulin and Pan, Xuran and Song, Shiji and Zhang, Hong and Huang, Gao and Wu, Cheng},
    booktitle = {Advances in Neural Information Processing Systems 32},
        pages = {12635--12644},
         year = {2019},
}
```

## Results

- Supervised image classification on ImageNet
<p align="center">
    <img src="ImageNet.png" height="284" width= "900">
</p>

- Complementing traditional data augmentation techniques
<p align="center">
    <img src="Complementary results.png" height="212" width= "900">
</p>

- Semi-supervised image classification on CIFAR & SVHN
<p align="center">
    <img src="Semi supervised learning.png" height="278" width= "900">
</p>

- Semantic segmentation on Cityscapes
<p align="center">
    <img src="Semantic Segmentation.png" height="250" width= "900">
</p>

- Object detection on MS COCO
<p align="center">
    <img src="Object detection.png" height="365" width= "900">
</p>

- Instance segmentation on MS COCO
<p align="center">
    <img src="Instance Segmentation.png" height="174" width= "900">
</p>

## Acknowledgment
Our code for semantic segmentation is mainly based on
[pytorch-segmentation-toolbox](https://github.com/speedinghzl/pytorch-segmentation-toolbox).

