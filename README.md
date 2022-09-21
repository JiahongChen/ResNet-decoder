# ResNet-decoder in PyTorch
ResNet decoder using transposed ResNet (ResNet-50, ResNet-101)

## Prerequisites

1. Install [Pytorch](https://pytorch.org/)
1. Install torchsummary: ```pip install torchsummary```

## Usage
Adjust Network structure in ```test.py```
* ResNet-50 encoder: 
```
import res_encoder as enc
netF = enc.ResNet(enc.Bottleneck, [3, 4, 6, 3])
```
* ResNet-50 decoder: 
```
import res_decoder as dec
netD = dec.ResNet(dec.Bottleneck, [3, 6, 4, 3])
```


Run 
```Python test.py```

## Network Structure for ResNet-50 encoder-decoder
```
Feature shape: torch.Size([2, 2048, 1, 1])
Reconstrusted image size: torch.Size([2, 3, 221, 221])
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 111, 111]           9,408
       BatchNorm2d-2         [-1, 64, 111, 111]             128
              ReLU-3         [-1, 64, 111, 111]               0
         MaxPool2d-4  [[-1, 64, 56, 56], [-1, 64, 56, 56]]               0
            Conv2d-5           [-1, 64, 56, 56]           4,096
       BatchNorm2d-6           [-1, 64, 56, 56]             128
              ReLU-7           [-1, 64, 56, 56]               0
            Conv2d-8           [-1, 64, 56, 56]          36,864
       BatchNorm2d-9           [-1, 64, 56, 56]             128
             ReLU-10           [-1, 64, 56, 56]               0
           Conv2d-11          [-1, 256, 56, 56]          16,384
      BatchNorm2d-12          [-1, 256, 56, 56]             512
           Conv2d-13          [-1, 256, 56, 56]          16,384
      BatchNorm2d-14          [-1, 256, 56, 56]             512
             ReLU-15          [-1, 256, 56, 56]               0
       Bottleneck-16          [-1, 256, 56, 56]               0
           Conv2d-17           [-1, 64, 56, 56]          16,384
      BatchNorm2d-18           [-1, 64, 56, 56]             128
             ReLU-19           [-1, 64, 56, 56]               0
           Conv2d-20           [-1, 64, 56, 56]          36,864
      BatchNorm2d-21           [-1, 64, 56, 56]             128
             ReLU-22           [-1, 64, 56, 56]               0
           Conv2d-23          [-1, 256, 56, 56]          16,384
      BatchNorm2d-24          [-1, 256, 56, 56]             512
             ReLU-25          [-1, 256, 56, 56]               0
       Bottleneck-26          [-1, 256, 56, 56]               0
           Conv2d-27           [-1, 64, 56, 56]          16,384
      BatchNorm2d-28           [-1, 64, 56, 56]             128
             ReLU-29           [-1, 64, 56, 56]               0
           Conv2d-30           [-1, 64, 56, 56]          36,864
      BatchNorm2d-31           [-1, 64, 56, 56]             128
             ReLU-32           [-1, 64, 56, 56]               0
           Conv2d-33          [-1, 256, 56, 56]          16,384
      BatchNorm2d-34          [-1, 256, 56, 56]             512
             ReLU-35          [-1, 256, 56, 56]               0
       Bottleneck-36          [-1, 256, 56, 56]               0
           Conv2d-37          [-1, 128, 56, 56]          32,768
      BatchNorm2d-38          [-1, 128, 56, 56]             256
             ReLU-39          [-1, 128, 56, 56]               0
           Conv2d-40          [-1, 128, 28, 28]         147,456
      BatchNorm2d-41          [-1, 128, 28, 28]             256
             ReLU-42          [-1, 128, 28, 28]               0
           Conv2d-43          [-1, 512, 28, 28]          65,536
      BatchNorm2d-44          [-1, 512, 28, 28]           1,024
           Conv2d-45          [-1, 512, 28, 28]         131,072
      BatchNorm2d-46          [-1, 512, 28, 28]           1,024
             ReLU-47          [-1, 512, 28, 28]               0
       Bottleneck-48          [-1, 512, 28, 28]               0
           Conv2d-49          [-1, 128, 28, 28]          65,536
      BatchNorm2d-50          [-1, 128, 28, 28]             256
             ReLU-51          [-1, 128, 28, 28]               0
           Conv2d-52          [-1, 128, 28, 28]         147,456
      BatchNorm2d-53          [-1, 128, 28, 28]             256
             ReLU-54          [-1, 128, 28, 28]               0
           Conv2d-55          [-1, 512, 28, 28]          65,536
      BatchNorm2d-56          [-1, 512, 28, 28]           1,024
             ReLU-57          [-1, 512, 28, 28]               0
       Bottleneck-58          [-1, 512, 28, 28]               0
           Conv2d-59          [-1, 128, 28, 28]          65,536
      BatchNorm2d-60          [-1, 128, 28, 28]             256
             ReLU-61          [-1, 128, 28, 28]               0
           Conv2d-62          [-1, 128, 28, 28]         147,456
      BatchNorm2d-63          [-1, 128, 28, 28]             256
             ReLU-64          [-1, 128, 28, 28]               0
           Conv2d-65          [-1, 512, 28, 28]          65,536
      BatchNorm2d-66          [-1, 512, 28, 28]           1,024
             ReLU-67          [-1, 512, 28, 28]               0
       Bottleneck-68          [-1, 512, 28, 28]               0
           Conv2d-69          [-1, 128, 28, 28]          65,536
      BatchNorm2d-70          [-1, 128, 28, 28]             256
             ReLU-71          [-1, 128, 28, 28]               0
           Conv2d-72          [-1, 128, 28, 28]         147,456
      BatchNorm2d-73          [-1, 128, 28, 28]             256
             ReLU-74          [-1, 128, 28, 28]               0
           Conv2d-75          [-1, 512, 28, 28]          65,536
      BatchNorm2d-76          [-1, 512, 28, 28]           1,024
             ReLU-77          [-1, 512, 28, 28]               0
       Bottleneck-78          [-1, 512, 28, 28]               0
           Conv2d-79          [-1, 256, 28, 28]         131,072
      BatchNorm2d-80          [-1, 256, 28, 28]             512
             ReLU-81          [-1, 256, 28, 28]               0
           Conv2d-82          [-1, 256, 14, 14]         589,824
      BatchNorm2d-83          [-1, 256, 14, 14]             512
             ReLU-84          [-1, 256, 14, 14]               0
           Conv2d-85         [-1, 1024, 14, 14]         262,144
      BatchNorm2d-86         [-1, 1024, 14, 14]           2,048
           Conv2d-87         [-1, 1024, 14, 14]         524,288
      BatchNorm2d-88         [-1, 1024, 14, 14]           2,048
             ReLU-89         [-1, 1024, 14, 14]               0
       Bottleneck-90         [-1, 1024, 14, 14]               0
           Conv2d-91          [-1, 256, 14, 14]         262,144
      BatchNorm2d-92          [-1, 256, 14, 14]             512
             ReLU-93          [-1, 256, 14, 14]               0
           Conv2d-94          [-1, 256, 14, 14]         589,824
      BatchNorm2d-95          [-1, 256, 14, 14]             512
             ReLU-96          [-1, 256, 14, 14]               0
           Conv2d-97         [-1, 1024, 14, 14]         262,144
      BatchNorm2d-98         [-1, 1024, 14, 14]           2,048
             ReLU-99         [-1, 1024, 14, 14]               0
      Bottleneck-100         [-1, 1024, 14, 14]               0
          Conv2d-101          [-1, 256, 14, 14]         262,144
     BatchNorm2d-102          [-1, 256, 14, 14]             512
            ReLU-103          [-1, 256, 14, 14]               0
          Conv2d-104          [-1, 256, 14, 14]         589,824
     BatchNorm2d-105          [-1, 256, 14, 14]             512
            ReLU-106          [-1, 256, 14, 14]               0
          Conv2d-107         [-1, 1024, 14, 14]         262,144
     BatchNorm2d-108         [-1, 1024, 14, 14]           2,048
            ReLU-109         [-1, 1024, 14, 14]               0
      Bottleneck-110         [-1, 1024, 14, 14]               0
          Conv2d-111          [-1, 256, 14, 14]         262,144
     BatchNorm2d-112          [-1, 256, 14, 14]             512
            ReLU-113          [-1, 256, 14, 14]               0
          Conv2d-114          [-1, 256, 14, 14]         589,824
     BatchNorm2d-115          [-1, 256, 14, 14]             512
            ReLU-116          [-1, 256, 14, 14]               0
          Conv2d-117         [-1, 1024, 14, 14]         262,144
     BatchNorm2d-118         [-1, 1024, 14, 14]           2,048
            ReLU-119         [-1, 1024, 14, 14]               0
      Bottleneck-120         [-1, 1024, 14, 14]               0
          Conv2d-121          [-1, 256, 14, 14]         262,144
     BatchNorm2d-122          [-1, 256, 14, 14]             512
            ReLU-123          [-1, 256, 14, 14]               0
          Conv2d-124          [-1, 256, 14, 14]         589,824
     BatchNorm2d-125          [-1, 256, 14, 14]             512
            ReLU-126          [-1, 256, 14, 14]               0
          Conv2d-127         [-1, 1024, 14, 14]         262,144
     BatchNorm2d-128         [-1, 1024, 14, 14]           2,048
            ReLU-129         [-1, 1024, 14, 14]               0
      Bottleneck-130         [-1, 1024, 14, 14]               0
          Conv2d-131          [-1, 256, 14, 14]         262,144
     BatchNorm2d-132          [-1, 256, 14, 14]             512
            ReLU-133          [-1, 256, 14, 14]               0
          Conv2d-134          [-1, 256, 14, 14]         589,824
     BatchNorm2d-135          [-1, 256, 14, 14]             512
            ReLU-136          [-1, 256, 14, 14]               0
          Conv2d-137         [-1, 1024, 14, 14]         262,144
     BatchNorm2d-138         [-1, 1024, 14, 14]           2,048
            ReLU-139         [-1, 1024, 14, 14]               0
      Bottleneck-140         [-1, 1024, 14, 14]               0
          Conv2d-141          [-1, 512, 14, 14]         524,288
     BatchNorm2d-142          [-1, 512, 14, 14]           1,024
            ReLU-143          [-1, 512, 14, 14]               0
          Conv2d-144            [-1, 512, 7, 7]       2,359,296
     BatchNorm2d-145            [-1, 512, 7, 7]           1,024
            ReLU-146            [-1, 512, 7, 7]               0
          Conv2d-147           [-1, 2048, 7, 7]       1,048,576
     BatchNorm2d-148           [-1, 2048, 7, 7]           4,096
          Conv2d-149           [-1, 2048, 7, 7]       2,097,152
     BatchNorm2d-150           [-1, 2048, 7, 7]           4,096
            ReLU-151           [-1, 2048, 7, 7]               0
      Bottleneck-152           [-1, 2048, 7, 7]               0
          Conv2d-153            [-1, 512, 7, 7]       1,048,576
     BatchNorm2d-154            [-1, 512, 7, 7]           1,024
            ReLU-155            [-1, 512, 7, 7]               0
          Conv2d-156            [-1, 512, 7, 7]       2,359,296
     BatchNorm2d-157            [-1, 512, 7, 7]           1,024
            ReLU-158            [-1, 512, 7, 7]               0
          Conv2d-159           [-1, 2048, 7, 7]       1,048,576
     BatchNorm2d-160           [-1, 2048, 7, 7]           4,096
            ReLU-161           [-1, 2048, 7, 7]               0
      Bottleneck-162           [-1, 2048, 7, 7]               0
          Conv2d-163            [-1, 512, 7, 7]       1,048,576
     BatchNorm2d-164            [-1, 512, 7, 7]           1,024
            ReLU-165            [-1, 512, 7, 7]               0
          Conv2d-166            [-1, 512, 7, 7]       2,359,296
     BatchNorm2d-167            [-1, 512, 7, 7]           1,024
            ReLU-168            [-1, 512, 7, 7]               0
          Conv2d-169           [-1, 2048, 7, 7]       1,048,576
     BatchNorm2d-170           [-1, 2048, 7, 7]           4,096
            ReLU-171           [-1, 2048, 7, 7]               0
      Bottleneck-172           [-1, 2048, 7, 7]               0
AdaptiveAvgPool2d-173           [-1, 2048, 1, 1]               0
================================================================
Total params: 23,508,032
Trainable params: 23,508,032
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.56
Forward/backward pass size (MB): 12131.31
Params size (MB): 89.68
Estimated Total Size (MB): 12221.54
----------------------------------------------------------------
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
          Upsample-1           [-1, 2048, 7, 7]               0
   ConvTranspose2d-2            [-1, 512, 7, 7]       1,048,576
       BatchNorm2d-3            [-1, 512, 7, 7]           1,024
              ReLU-4            [-1, 512, 7, 7]               0
   ConvTranspose2d-5            [-1, 512, 7, 7]       2,359,296
       BatchNorm2d-6            [-1, 512, 7, 7]           1,024
              ReLU-7            [-1, 512, 7, 7]               0
   ConvTranspose2d-8           [-1, 2048, 7, 7]       1,048,576
       BatchNorm2d-9           [-1, 2048, 7, 7]           4,096
             ReLU-10           [-1, 2048, 7, 7]               0
       Bottleneck-11           [-1, 2048, 7, 7]               0
  ConvTranspose2d-12            [-1, 512, 7, 7]       1,048,576
      BatchNorm2d-13            [-1, 512, 7, 7]           1,024
             ReLU-14            [-1, 512, 7, 7]               0
  ConvTranspose2d-15            [-1, 512, 7, 7]       2,359,296
      BatchNorm2d-16            [-1, 512, 7, 7]           1,024
             ReLU-17            [-1, 512, 7, 7]               0
  ConvTranspose2d-18           [-1, 2048, 7, 7]       1,048,576
      BatchNorm2d-19           [-1, 2048, 7, 7]           4,096
             ReLU-20           [-1, 2048, 7, 7]               0
       Bottleneck-21           [-1, 2048, 7, 7]               0
  ConvTranspose2d-22            [-1, 512, 7, 7]       1,048,576
      BatchNorm2d-23            [-1, 512, 7, 7]           1,024
             ReLU-24            [-1, 512, 7, 7]               0
  ConvTranspose2d-25          [-1, 512, 14, 14]       2,359,296
      BatchNorm2d-26          [-1, 512, 14, 14]           1,024
             ReLU-27          [-1, 512, 14, 14]               0
  ConvTranspose2d-28         [-1, 1024, 14, 14]         524,288
      BatchNorm2d-29         [-1, 1024, 14, 14]           2,048
  ConvTranspose2d-30         [-1, 1024, 14, 14]       2,097,152
      BatchNorm2d-31         [-1, 1024, 14, 14]           2,048
             ReLU-32         [-1, 1024, 14, 14]               0
       Bottleneck-33         [-1, 1024, 14, 14]               0
  ConvTranspose2d-34          [-1, 256, 14, 14]         262,144
      BatchNorm2d-35          [-1, 256, 14, 14]             512
             ReLU-36          [-1, 256, 14, 14]               0
  ConvTranspose2d-37          [-1, 256, 14, 14]         589,824
      BatchNorm2d-38          [-1, 256, 14, 14]             512
             ReLU-39          [-1, 256, 14, 14]               0
  ConvTranspose2d-40         [-1, 1024, 14, 14]         262,144
      BatchNorm2d-41         [-1, 1024, 14, 14]           2,048
             ReLU-42         [-1, 1024, 14, 14]               0
       Bottleneck-43         [-1, 1024, 14, 14]               0
  ConvTranspose2d-44          [-1, 256, 14, 14]         262,144
      BatchNorm2d-45          [-1, 256, 14, 14]             512
             ReLU-46          [-1, 256, 14, 14]               0
  ConvTranspose2d-47          [-1, 256, 14, 14]         589,824
      BatchNorm2d-48          [-1, 256, 14, 14]             512
             ReLU-49          [-1, 256, 14, 14]               0
  ConvTranspose2d-50         [-1, 1024, 14, 14]         262,144
      BatchNorm2d-51         [-1, 1024, 14, 14]           2,048
             ReLU-52         [-1, 1024, 14, 14]               0
       Bottleneck-53         [-1, 1024, 14, 14]               0
  ConvTranspose2d-54          [-1, 256, 14, 14]         262,144
      BatchNorm2d-55          [-1, 256, 14, 14]             512
             ReLU-56          [-1, 256, 14, 14]               0
  ConvTranspose2d-57          [-1, 256, 14, 14]         589,824
      BatchNorm2d-58          [-1, 256, 14, 14]             512
             ReLU-59          [-1, 256, 14, 14]               0
  ConvTranspose2d-60         [-1, 1024, 14, 14]         262,144
      BatchNorm2d-61         [-1, 1024, 14, 14]           2,048
             ReLU-62         [-1, 1024, 14, 14]               0
       Bottleneck-63         [-1, 1024, 14, 14]               0
  ConvTranspose2d-64          [-1, 256, 14, 14]         262,144
      BatchNorm2d-65          [-1, 256, 14, 14]             512
             ReLU-66          [-1, 256, 14, 14]               0
  ConvTranspose2d-67          [-1, 256, 28, 28]         589,824
      BatchNorm2d-68          [-1, 256, 28, 28]             512
             ReLU-69          [-1, 256, 28, 28]               0
  ConvTranspose2d-70          [-1, 512, 28, 28]         131,072
      BatchNorm2d-71          [-1, 512, 28, 28]           1,024
  ConvTranspose2d-72          [-1, 512, 28, 28]         524,288
      BatchNorm2d-73          [-1, 512, 28, 28]           1,024
             ReLU-74          [-1, 512, 28, 28]               0
       Bottleneck-75          [-1, 512, 28, 28]               0
  ConvTranspose2d-76          [-1, 128, 28, 28]          65,536
      BatchNorm2d-77          [-1, 128, 28, 28]             256
             ReLU-78          [-1, 128, 28, 28]               0
  ConvTranspose2d-79          [-1, 128, 28, 28]         147,456
      BatchNorm2d-80          [-1, 128, 28, 28]             256
             ReLU-81          [-1, 128, 28, 28]               0
  ConvTranspose2d-82          [-1, 512, 28, 28]          65,536
      BatchNorm2d-83          [-1, 512, 28, 28]           1,024
             ReLU-84          [-1, 512, 28, 28]               0
       Bottleneck-85          [-1, 512, 28, 28]               0
  ConvTranspose2d-86          [-1, 128, 28, 28]          65,536
      BatchNorm2d-87          [-1, 128, 28, 28]             256
             ReLU-88          [-1, 128, 28, 28]               0
  ConvTranspose2d-89          [-1, 128, 28, 28]         147,456
      BatchNorm2d-90          [-1, 128, 28, 28]             256
             ReLU-91          [-1, 128, 28, 28]               0
  ConvTranspose2d-92          [-1, 512, 28, 28]          65,536
      BatchNorm2d-93          [-1, 512, 28, 28]           1,024
             ReLU-94          [-1, 512, 28, 28]               0
       Bottleneck-95          [-1, 512, 28, 28]               0
  ConvTranspose2d-96          [-1, 128, 28, 28]          65,536
      BatchNorm2d-97          [-1, 128, 28, 28]             256
             ReLU-98          [-1, 128, 28, 28]               0
  ConvTranspose2d-99          [-1, 128, 28, 28]         147,456
     BatchNorm2d-100          [-1, 128, 28, 28]             256
            ReLU-101          [-1, 128, 28, 28]               0
 ConvTranspose2d-102          [-1, 512, 28, 28]          65,536
     BatchNorm2d-103          [-1, 512, 28, 28]           1,024
            ReLU-104          [-1, 512, 28, 28]               0
      Bottleneck-105          [-1, 512, 28, 28]               0
 ConvTranspose2d-106          [-1, 128, 28, 28]          65,536
     BatchNorm2d-107          [-1, 128, 28, 28]             256
            ReLU-108          [-1, 128, 28, 28]               0
 ConvTranspose2d-109          [-1, 128, 28, 28]         147,456
     BatchNorm2d-110          [-1, 128, 28, 28]             256
            ReLU-111          [-1, 128, 28, 28]               0
 ConvTranspose2d-112          [-1, 512, 28, 28]          65,536
     BatchNorm2d-113          [-1, 512, 28, 28]           1,024
            ReLU-114          [-1, 512, 28, 28]               0
      Bottleneck-115          [-1, 512, 28, 28]               0
 ConvTranspose2d-116          [-1, 128, 28, 28]          65,536
     BatchNorm2d-117          [-1, 128, 28, 28]             256
            ReLU-118          [-1, 128, 28, 28]               0
 ConvTranspose2d-119          [-1, 128, 28, 28]         147,456
     BatchNorm2d-120          [-1, 128, 28, 28]             256
            ReLU-121          [-1, 128, 28, 28]               0
 ConvTranspose2d-122          [-1, 512, 28, 28]          65,536
     BatchNorm2d-123          [-1, 512, 28, 28]           1,024
            ReLU-124          [-1, 512, 28, 28]               0
      Bottleneck-125          [-1, 512, 28, 28]               0
 ConvTranspose2d-126          [-1, 128, 28, 28]          65,536
     BatchNorm2d-127          [-1, 128, 28, 28]             256
            ReLU-128          [-1, 128, 28, 28]               0
 ConvTranspose2d-129          [-1, 128, 56, 56]         147,456
     BatchNorm2d-130          [-1, 128, 56, 56]             256
            ReLU-131          [-1, 128, 56, 56]               0
 ConvTranspose2d-132          [-1, 256, 56, 56]          32,768
     BatchNorm2d-133          [-1, 256, 56, 56]             512
 ConvTranspose2d-134          [-1, 256, 56, 56]         131,072
     BatchNorm2d-135          [-1, 256, 56, 56]             512
            ReLU-136          [-1, 256, 56, 56]               0
      Bottleneck-137          [-1, 256, 56, 56]               0
 ConvTranspose2d-138           [-1, 64, 56, 56]          16,384
     BatchNorm2d-139           [-1, 64, 56, 56]             128
            ReLU-140           [-1, 64, 56, 56]               0
 ConvTranspose2d-141           [-1, 64, 56, 56]          36,864
     BatchNorm2d-142           [-1, 64, 56, 56]             128
            ReLU-143           [-1, 64, 56, 56]               0
 ConvTranspose2d-144          [-1, 256, 56, 56]          16,384
     BatchNorm2d-145          [-1, 256, 56, 56]             512
            ReLU-146          [-1, 256, 56, 56]               0
      Bottleneck-147          [-1, 256, 56, 56]               0
 ConvTranspose2d-148           [-1, 64, 56, 56]          16,384
     BatchNorm2d-149           [-1, 64, 56, 56]             128
            ReLU-150           [-1, 64, 56, 56]               0
 ConvTranspose2d-151           [-1, 64, 56, 56]          36,864
     BatchNorm2d-152           [-1, 64, 56, 56]             128
            ReLU-153           [-1, 64, 56, 56]               0
 ConvTranspose2d-154          [-1, 256, 56, 56]          16,384
     BatchNorm2d-155          [-1, 256, 56, 56]             512
            ReLU-156          [-1, 256, 56, 56]               0
      Bottleneck-157          [-1, 256, 56, 56]               0
 ConvTranspose2d-158           [-1, 64, 56, 56]          16,384
     BatchNorm2d-159           [-1, 64, 56, 56]             128
            ReLU-160           [-1, 64, 56, 56]               0
 ConvTranspose2d-161           [-1, 64, 56, 56]          36,864
     BatchNorm2d-162           [-1, 64, 56, 56]             128
            ReLU-163           [-1, 64, 56, 56]               0
 ConvTranspose2d-164           [-1, 64, 56, 56]           4,096
     BatchNorm2d-165           [-1, 64, 56, 56]             128
 ConvTranspose2d-166           [-1, 64, 56, 56]          16,384
     BatchNorm2d-167           [-1, 64, 56, 56]             128
            ReLU-168           [-1, 64, 56, 56]               0
      Bottleneck-169           [-1, 64, 56, 56]               0
================================================================
Total params: 21,816,320
Trainable params: 21,816,320
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 288.83
Params size (MB): 83.22
Estimated Total Size (MB): 372.06
----------------------------------------------------------------
```

# Reference
* The backbone network is modified from the [official PyTorch ResNet package](https://github.com/pytorch/vision/blob/a4f53308b2d0f1aa9191686e326f45c26053f686/torchvision/models/resnet.py)
* If you find this project useful please cite the original papers:
    * [ResNet-50](https://arxiv.org/pdf/1512.03385.pdf)
    * [ResNet-101](https://arxiv.org/pdf/1512.03385.pdf)
