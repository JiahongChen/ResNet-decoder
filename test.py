# reference to https://github.com/pytorch/vision/blob/a4f53308b2d0f1aa9191686e326f45c26053f686/torchvision/models/resnet.py#L288
from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

from torchsummary import summary

import res_encoder as enc
import res_decoder as dec

if __name__ == "__main__":
	netF = enc.ResNet(enc.Bottleneck, [3, 4, 23, 3], return_indices=True)
	state_dict = torch.load('model/resnet101.pth') # https://download.pytorch.org/models/resnet101-63fe2227.pth
	# state_dict = torch.load('model/resnet50.pth') # https://download.pytorch.org/models/resnet50-0676ba61.pth
	netF.load_state_dict(state_dict)


	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	netF.to(device)
	test_input = torch.rand(2, 3, 221, 221).to(device)
	# input_size = (3, 224, 224)
	# test_input = [torch.rand(2, *input_size).type(torch.FloatTensor).to(device=device)]
	out, indices = netF(test_input)

	print('Feature shape:', out.shape)

	netD = dec.ResNet(dec.Bottleneck, [3, 23, 4, 3])
	netD.to(device)
	rec = netD(out, indices)

	print('Reconstrusted image size:', rec.shape)

	# summary(netD, [(2048, 1, 1), (64, 56, 56)])
	summary(netF, (3, 221, 221))
	summary(netD, (2048, 1, 1))


