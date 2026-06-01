#!/usr/bin/env python

from __future__ import print_function

##################################################
###          MODULE IMPORT
##################################################
# - STANDARD MODULES
import sys
import os
import random
import numpy as np
from typing import List, Tuple, Sequence, Union, Optional

# - TORCH
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

PILImageTypes = tuple()  # filled lazily to avoid hard PIL dependency at import
try:
	from PIL import Image
	PILImageTypes = (Image.Image,)
except Exception:
	pass

##########################################
##    FlippingTransform
##########################################
class FlippingTransform(torch.nn.Module):
	"""Flipping: lr, ud or nothing"""

	def __init__(self):
		super().__init__()

	def forward(self, img):
		op= random.choice([1,2,3])
		if op==1:
			return TF.hflip(img)
		elif op==2:
			return TF.vflip(img)
		else:
			return img

##########################################
##     Rotate90 transform
##########################################
class Rotate90Transform(torch.nn.Module):
	"""Rotate by one of the given angles: 90, 270, """

	def __init__(self):
		super().__init__()

	def forward(self, img):
		op= random.choice([1,2,3,4])
		if op==1:
			return TF.rotate(img, 90)
		elif op==2:
			return TF.rotate(img, 180)
		elif op==3:
			return TF.rotate(img, 270)
		elif op==4:
			return img
			
##########################################
##     RandomCenterCrop transform
##########################################
def _center_crop_params(h: int, w: int, frac: float) -> Tuple[int, int, int, int]:
	"""Return (top, left, new_h, new_w) for a center crop with the given scale fraction."""
	new_h = max(1, int(round(h * frac)))
	new_w = max(1, int(round(w * frac)))
	top = max(0, (h - new_h) // 2)
	left = max(0, (w - new_w) // 2)
	return top, left, new_h, new_w

def _is_tensor_image(x: torch.Tensor) -> bool:
	# Accept (C,H,W) or (H,W) tensors
	return torch.is_tensor(x) and x.ndim in (2, 3)
    
def _resize_image(img, size: int):
	if isinstance(img, PILImageTypes):
		return TF.resize(img, size=(size, size), interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
	if _is_tensor_image(img):
		return TF.resize(img, size=(size, size), interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
	raise TypeError(f"Unsupported image type: {type(img)}")
    		
def _crop_image(img, top: int, left: int, height: int, width: int):
	# Works for PIL or torch Tensor
	if isinstance(img, PILImageTypes):
		return TF.crop(img, top, left, height, width)
	if _is_tensor_image(img):
		# torchscript-compatible crop
		return img[..., top:top+height, left:left+width]
	raise TypeError(f"Unsupported image type: {type(img)}")    		
    		
class RandomCenterCrop(torch.nn.Module):
	"""
		Center crop with a random scale fraction in [min_frac, max_frac], keeping aspect ratio.
		Optionally resizes back to a square 'output_size'.
		Works on a single image (PIL or Tensor (C,H,W)).
	"""
	def __init__(
		self,
		min_frac: float = 0.7,
		max_frac: float = 1.0,
		output_size: Optional[int] = None,
		generator: Optional[torch.Generator] = None,
	):
		super().__init__()
		assert 0.0 < min_frac <= max_frac <= 1.0
		self.min_frac = float(min_frac)
		self.max_frac = float(max_frac)
		self.output_size = output_size
		self.generator = generator  # for reproducibility if you pass a seeded generator

	def _rand_uniform(self) -> float:
		if self.generator is None:
			return random.random()
		# Torch generator path (stable across Python processes if seeded)
		return float(torch.rand((), generator=self.generator).item())

	def forward(self, img: Union[torch.Tensor, "Image.Image"]):
		if isinstance(img, PILImageTypes):
			w, h = img.size
		elif _is_tensor_image(img):
			# (C,H,W)
			h, w = img.shape[-2], img.shape[-1]
		else:
			raise TypeError(f"Unsupported image type: {type(img)}")

		frac = self.min_frac + (self.max_frac - self.min_frac) * self._rand_uniform()
		top, left, new_h, new_w = _center_crop_params(h, w, frac)

		img = _crop_image(img, top, left, new_h, new_w)
		if self.output_size is not None:
			img = _resize_image(img, self.output_size)

		return img
		
##########################################
##     MinMaxNormalization transform
##########################################
import torch
import torchvision.transforms.functional as F
from typing import Union, Optional

class MinMaxNormalization(torch.nn.Module):
	"""
		Min-max normalization scaling an image to [norm_min, norm_max].
		Works on a single image (PIL or Tensor (C,H,W)).
		Note: Output is always a torch.Tensor, as arbitrary float ranges 
		cannot be reliably represented by standard PIL RGB images.
	"""
	def __init__(
		self,
		norm_min: float = 0.0,
		norm_max: float = 255.0,
		eps: float = 1e-8,
		sanitize_tensor: bool = True
	):
		super().__init__()
  	assert norm_min < norm_max, "norm_min must be strictly less than norm_max."
		self.norm_min = float(norm_min)
		self.norm_max = float(norm_max)
		self.eps = float(eps)
		self.sanitize_tensor = sanitize_tensor

	def forward(self, img: Union[torch.Tensor, "Image.Image"]) -> torch.Tensor:
		# 1. Type handling and conversion
		if isinstance(img, PILImageTypes):
			img = F.to_tensor(img)
		elif _is_tensor_image(img):
			if not img.is_floating_point():
				img = img.float()
		else:
			raise TypeError(f"Unsupported image type: {type(img)}")

		# 2. Sanitize the tensor to protect against NaN and Inf
		if self.sanitize_tensor:
			# Replaces NaN with 0.0, +Inf with max finite, -Inf with min finite
			# This guarantees img.min() and img.max() will return usable numbers
			img = torch.nan_to_num(img)

		# 3. Calculate global min and max
		img_min = img.min()
		img_max = img.max()

		# 4. Normalize to [0.0, 1.0] (eps prevents ZeroDivisionError on flat images)
		img_normalized = (img - img_min) / (img_max - img_min + self.eps)

		# 5. Scale to [norm_min, norm_max]
		img_scaled = img_normalized * (self.norm_max - self.norm_min) + self.norm_min

		return img_scaled

