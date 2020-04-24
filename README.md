# Implicit Semantic Data Augmentation for Deep Networks **(NeurIPS 2019)**

Code for the paper 'Implicit Semantic Data Augmentation for Deep Networks' (https://arxiv.org/abs/1909.12220)

In this paper, we propose a novel implicit semantic data augmentation (ISDA) approach to complement traditional augmentation techniques like ﬂipping, translation or rotation.
ISDA consistently improves the generalization performance of popular deep networks on supervised & semi-supervised image classification, semantic segmentation, object detection and instance segmentation.

<p align="center">
    <img src="ISDA-overview.png" height="343" width= "1000">
</p>


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
    <img src="ImageNet.png" height="316" width= "1000">
</p>

- Complementing traditional data augmentation techniques
<p align="center">
    <img src="Complementary results.png" height="235" width= "1000">
</p>

- Semi-supervised image classification on CIFAR & SVHN
<p align="center">
    <img src="Semi supervised learning.png" height="309" width= "1000">
</p>

- Semantic segmentation on Cityscapes
<p align="center">
    <img src="Semantic Segmentation.png" height="278" width= "1000">
</p>


