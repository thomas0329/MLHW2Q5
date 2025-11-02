# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
	def __init__(self, channels: int, se: bool = False, se_reduction: int = 8):
		super().__init__()
		self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(channels)
		self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(channels)
		self.se = se
		if se:
			self.se_fc1 = nn.Linear(channels, channels // se_reduction)
			self.se_fc2 = nn.Linear(channels // se_reduction, channels)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		identity = x
		x = F.relu(self.bn1(self.conv1(x)), inplace=True)
		x = self.bn2(self.conv2(x))
		if self.se:
			w = F.adaptive_avg_pool2d(x, output_size=1).flatten(1)
			w = F.relu(self.se_fc1(w), inplace=True)
			w = torch.sigmoid(self.se_fc2(w))
			x = x * w.unsqueeze(-1).unsqueeze(-1)
		x = F.relu(x + identity, inplace=True)
		return x


class MoveEncoderCNN(nn.Module):
	def __init__(self, in_channels: int = 4, base_channels: int = 64, num_blocks: int = 4, se: bool = False, out_dim: int = 256):
		super().__init__()
		self.stem = nn.Sequential(
			nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(base_channels),
			nn.ReLU(inplace=True),
		)
		self.blocks = nn.Sequential(*[ResidualBlock(base_channels, se=se) for _ in range(num_blocks)])
		self.head = nn.Linear(base_channels, out_dim)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: [B, C, 19, 19]
		x = self.stem(x)
		x = self.blocks(x)
		x = F.adaptive_avg_pool2d(x, output_size=1).flatten(1)
		x = self.head(x)
		return x
