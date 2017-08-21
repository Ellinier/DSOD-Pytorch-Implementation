import torch

import torch.nn as nn

import torch.nn.functional as F

import math

from multibox_layer import MultiBoxLayer

# class L2Norm(nn.Module):
# 	def __init__(self, nChannels, scale):
# 		super(L2Norm, self).__init__()
# 		self.nChannels = nChannels
# 		self.gamma = scale or None
# 		self.eps = 1e-10
# 		self.weight = nn.Parameter(torch.Tensor(self.nChannels))
# 		self.reset_parameters()

# 	def reset_parameters(self):
# 		init.constant(self.weight, self.gamma)

# 	def forward(self, x):
# 		norm = x.pow(2).sum(1).sqrt() + self.eps
# 		x = x/norm.expand_as(x)
# 		out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
# 		return out

class StemLayer(nn.Module):
	def __init__(self, nChannels, nOutChannels, stride):
		super(StemLayer, self).__init__()
		self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(nOutChannels)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		return out

class SingleLayer(nn.Module):
	def __init__(self, nChannels, nOutChannels, kernel_size, stride, padding, dropout=0):
		super(SingleLayer, self).__init__()
		self.bn1 = nn.BatchNorm2d(nChannels)
		self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

	def forward(self, x):
		out = self.conv1(F.relu(self.bn1(x)))
		# if dropout > 0:
		# 		out = F.dropout2d(out, dropout)
		return out

class SingleLayer2(nn.Module):
	def __init__(self, nChannels, nOutChannels):
		super(SingleLayer2, self).__init__()
		self.bn1 = nn.BatchNorm2d(nChannels)
		self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn2 = nn.BatchNorm2d(nOutChannels)
		self.conv2 = nn.Conv2d(nOutChannels, nOutChannels, kernel_size=3, stride=2, padding=1, bias=False)
		self.bn3 = nn.BatchNorm2d(nChannels)
		self.conv3 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, stride=1, padding=0, bias=False)

	def forward(self, x):
		out1 = self.conv1(F.relu(self.bn1(x)))
		out1 = self.conv2(F.relu(self.bn2(out1)))
		out2 = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
		out2 = self.conv3(F.relu(self.bn3(out2)))
		out = torch.cat((out1, out2), 1)
		return out

class LastLayer(nn.Module):
	def __init__(self, nChannels, nOutChannels):
		super(LastLayer, self).__init__()
		self.bn1 = nn.BatchNorm2d(nChannels)
		self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn2 = nn.BatchNorm2d(nOutChannels)
		self.conv2 = nn.Conv2d(nOutChannels, nOutChannels, kernel_size=3, stride=2, padding=0, bias=False)
		self.bn3 = nn.BatchNorm2d(nChannels)
		self.conv3 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, stride=1, padding=0, bias=False)

	def forward(self, x):
		out1 = self.conv1(F.relu(self.bn1(x)))
		out1 = self.conv2(F.relu(self.bn2(out1)))
		out2 = F.max_pool2d(x, kernel_size=2, stride=2)
		out2 = self.conv3(F.relu(self.bn3(out2)))
		out = torch.cat((out1, out2), 1)
		return out

class Bottleneck(nn.Module):
	def __init__(self, nChannels, growthRate):
		super(Bottleneck, self).__init__()

		interChannels = 4*growthRate
		self.bn1 = nn.BatchNorm2d(nChannels)
		self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1, bias=False)
		self.bn2 = nn.BatchNorm2d(interChannels)
		self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3, padding=1, bias=False)

	def forward(self, x):
		out = self.conv1(F.relu(self.bn1(x)))
		out = self.conv2(F.relu(self.bn2(out)))
		out = torch.cat((x, out), 1)
		return out

class Transition(nn.Module):
	def __init__(self, nChannels, nOutChannels):
		super(Transition, self).__init__()
		self.bn1 = nn.BatchNorm2d(nChannels)
		self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, bias=False)

	def forward(self, x):
		out = self.conv1(F.relu(self.bn1(x)))
		out = F.max_pool2d(out, kernel_size=2, stride=2, ceil_mode=True)
		return out

# class Transition3x3(nn.Module):
# 	def __init__(self, nChannels, nOutChannels):
# 		super(Transition3x3, self).__init__()
# 		self.bn1 = nn.BatchNorm2d(nChannels)
# 		self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=3, bias=False)

# 	def forward(self, x):
# 		out = self.conv1(F.relu(self.bn1(x)))
# 		return out

class Transition_w_o_pooling(nn.Module):
	def __init__(self, nChannels, nOutChannels):
		super(Transition_w_o_pooling, self).__init__()
		self.bn1 = nn.BatchNorm2d(nChannels)
		self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, bias=False)

	def forward(self, x):
		out = self.conv1(F.relu(self.bn1(x)))
		return out


class DSOD(nn.Module):
	def __init__(self, growthRate, reduction):
		super(DSOD, self).__init__()

		# stem
		self.conv1 = StemLayer(3, 64, stride=2)
		self.conv2 = StemLayer(64, 64, stride=1)
		self.conv3 = StemLayer(64, 128, stride=1)
		self.dense1 = self._make_dense(128, growthRate, 6)
		nChannels = 128+6*growthRate
		nOutChannels = int(math.floor(nChannels*reduction))
		self.trans1 = Transition(nChannels, nOutChannels)

		nChannels = nOutChannels
		self.dense2 = self._make_dense(nChannels, growthRate, 8)
		nChannels += 8*growthRate
		nOutChannels1 = int(math.floor(nChannels*reduction))
		self.trans_wo = Transition_w_o_pooling(nChannels, nOutChannels1)
		# self.trans2 = Transition(nChannels, nOutChannels)
		# self.First = self.trans_wo

		nChannels = nOutChannels1
		self.dense3 = self._make_dense(nChannels, growthRate, 8)
		nChannels += 8*growthRate
		nOutChannels = int(math.floor(nChannels*reduction))
		self.trans_wo1 = Transition_w_o_pooling(nChannels, nOutChannels)

		nChannels = nOutChannels
		self.dense4 = self._make_dense(nChannels, growthRate, 8)
		nChannels += 8*growthRate
		# nOutChannels = int(math.floor(nChannels*reduction))
		nOutChannels = 256
		self.trans_wo2 = Transition_w_o_pooling(nChannels, nOutChannels)

		nChannels = nOutChannels1
		nOutChannels = 256
		self.conv4 = SingleLayer(nChannels, nOutChannels, kernel_size=1, stride=1, padding=0)

		# self.second = torch.cat((self.First, self.conv4), 1)

		# addExtraLyers
		nChannels = nOutChannels*2
		self.third = SingleLayer2(nChannels, 256)
		nChannels = 256*2
		self.forth = SingleLayer2(nChannels, 128)
		nChannels = 128*2
		self.fifith = SingleLayer2(nChannels, 128)
		nChannels = 128*2
		# self.sixth = SingleLayer2(nChannels, 128)
		self.sixth = LastLayer(nChannels, 128)

		# multibox layer
		self.multibox = MultiBoxLayer()

	def _make_dense(self, nChannels, growthRate, nDenseBlocks):
		layers = []
		for i in range(int(nDenseBlocks)):
			layers.append(Bottleneck(nChannels, growthRate))
			nChannels += growthRate
		return nn.Sequential(*layers)


	def forward(self, x):
		Out = []	
		out = self.conv1(x)
		out = self.conv2(out)
		out = self.conv3(out)
		out = F.max_pool2d(out, kernel_size=2, stride=2)

		out = self.trans1(self.dense1(out))
		out = self.trans_wo(self.dense2(out))
		First = out
		Out.append(First)
		out = F.max_pool2d(out, kernel_size=2, stride=2)
		# out = self.trans2(self.dense2(out))
		out = self.trans_wo1(self.dense3(out))
		out = self.trans_wo2(self.dense4(out))

		f_first = F.max_pool2d(First, kernel_size=2, stride=2)
		f_first = self.conv4(f_first)
		out = torch.cat((out, f_first), 1)
		Second = out
		Out.append(Second)

		Third = self.third(Second)
		Out.append(Third)
		Forth = self.forth(Third)
		Out.append(Forth)
		Fifth = self.fifith(Forth)
		Out.append(Fifth)
		Sixth = self.sixth(Fifth)
		Out.append(Sixth)

		loc_preds, conf_preds = self.multibox(Out)
		return loc_preds, conf_preds
