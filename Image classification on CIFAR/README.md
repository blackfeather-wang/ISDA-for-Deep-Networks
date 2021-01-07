# Image Classification on CIFAR

## Requirements
- python 3.5.4
- pytorch 1.0.1
- torchvision 0.2.2


## Run

Train Wide-ResNet-28-10 on CIFAR-10 / 100 with ISDA

```
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model wideresnet --layers 28 --widen-factor 10 --lambda_0 0.5 --droprate 0.3
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar100 --model wideresnet --layers 28 --widen-factor 10 --lambda_0 0.5 --droprate 0.3

```

Train Wide-ResNet-28-10 on CIFAR-10 / 100 with ISDA and AutoAugment

```
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model wideresnet --layers 28 --widen-factor 10 --lambda_0 0.5 --droprate 0.3 --autoaugment
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar100 --model wideresnet --layers 28 --widen-factor 10 --lambda_0 0.5 --droprate 0.3 --autoaugment

```


Train Shake-Shake(26, 2x112d) on CIFAR-10 / 100 with ISDA and AutoAugment

```
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model shake_shake --widen-factor 112 --lambda_0 0.5 --cos_lr --autoaugment
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar100 --model shake_shake --widen-factor 112 --lambda_0 0.5 --cos_lr --autoaugment

```

## Results

<p align="center">
    <img src="ISDA-cifar-1.png" height="414" width= "700">
</p>

<p align="center">
    <img src="ISDA-cifar-2.png" height="221" width= "700">
</p>



## Usage for Other Models

To apply ISDA to other models, the final fully connected layer needs to be explicitly defined:

```python
class Full_layer(torch.nn.Module):
    '''explicitly define the full connected layer'''

    def __init__(self, feature_num, class_num):
        super(Full_layer, self).__init__()
        self.class_num = class_num
        self.fc = nn.Linear(feature_num, class_num)

    def forward(self, x):
        x = self.fc(x)
        return x

fc = Full_layer(model.feature_num, class_num)
# model.feature_num is the dimension of deep features output by the model.
```

The model needs to output deep features instead of inference results:

```python
optimizer = torch.optim.SGD([{'params': model.parameters()},
                             {'params': fc.parameters()}],
                             ......)
......
from ISDA import ISDALoss
isda_criterion = ISDALoss(model.feature_num, class_num).cuda()
......
ratio = args.lambda_0 * (epoch / (training_configurations[args.model]['epochs']))
loss, output = criterion(model, fc, input_var, target_var, ratio)
```




## Argument Description

--dataset : cifar10 or cifar100

--model : deep networks to be trained, choose from {resnet, wideresnet, resnext, se_resnet, se_wideresnet, densenet_bc, shake_shake, shake_pyramidnet}

--layers : total number of layers

--lambda_0 : hyper-parameter \lambda_0 for ISDA (we recommend 0.5 or 0.25 for naive implementation.)

--droprate : specify the dropout rate

--widen-factor : widen factor for Wide-ResNet and Shake-Shake

--cardinality : cardinality for ResNeXt

--growth-rate, --compression-rate, --bn-size : hyper-parameters for DenseNet-BC

--alpha : alpha for PyramidNet

--autoaugment : to apply AutoAugment with searched policy on CIFAR

--cutout : to apply Cutout augmentation

--cos_lr : to use cosine learning rate schedule


## References

This repo is based on the codes from https://github.com/xternalz/WideResNet-pytorch

1. ResNet References
    - (ResNet) Deep Residual Learning for Image Recognition
      - Paper : https://arxiv.org/abs/1512.03385
    - (ResNet) Identity Mappings in Deep Residual Networks
      - Paper : https://arxiv.org/abs/1603.05027
    - Codes
      - https://github.com/osmr/imgclsmob/tree/master/pytorch/pytorchcv/models
2. (PyramidNet) Deep Pyramidal Residual Networks
    - Paper : https://arxiv.org/abs/1610.02915
    - Code : https://github.com/dyhan0920/PyramidNet-PyTorch
3. (Wide-ResNet)
    - Paper : http://arxiv.org/abs/1605.07146
4. (ResNeXt) Aggregated Residual Transformations for Deep Neural Networks
    - Paper : https://arxiv.org/pdf/1611.05431.pdf
    - Code : https://github.com/D-X-Y/ResNeXt-DenseNet
5. (SE-ResNet) Squeeze-and-Excitation Networks
    - Paper : https://arxiv.org/pdf/1709.01507.pdf
    - Code : https://github.com/moskomule/senet.pytorch
6. (DenseNet-BC) Densely Connected Convolutional Networks
    - Paper : https://arxiv.org/pdf/1608.06993.pdf
    - Code : https://github.com/bamos/densenet.pytorch
7. Shake-Shake
    - Paper : https://arxiv.org/pdf/1705.07485.pdf
    - Code : https://github.com/owruby/shake-shake_pytorch
8. ShakeDrop Regularization for Deep Residual Learning
    - Paper : https://arxiv.org/abs/1802.02375
    - Code : https://github.com/owruby/shake-drop_pytorch
9. AutoAugment
    - Paper : http://openaccess.thecvf.com/content_CVPR_2019/papers/Cubuk_AutoAugment_Learning_Augmentation_Strategies_From_Data_CVPR_2019_paper.pdf
    - Code : https://github.com/DeepVoltaire/AutoAugment
10. Cutout
    - Paper : https://arxiv.org/pdf/1708.04552.pdf
    - Code : https://github.com/uoguelph-mlrg/Cutout
