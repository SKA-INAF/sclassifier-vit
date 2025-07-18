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

# - PIL
import PIL
from PIL import Image

# - SKLEARN
from sklearn.preprocessing import MultiLabelBinarizer

# - TORCH
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T

# - SCLASSIFIER-VIT
from sclassifier_vit.utils import *

######################################
###      CLASS LABEL SCHEMA
######################################
def get_multi_label_target_maps(schema="morph_tags", skip_first_class=False):
	""" Return multi-label classifier target maps """

	if schema=="morph_tags":
		if skip_first_class:
			id2target= {
				0: -1, # background
				1: 0, # radio-galaxy
				2: 1, # extended
				3: 2, # diffuse
				4: 3, # diffuse-large
				5: 4, # artefact
			}
			
			id2label= {
				0: "RADIO-GALAXY",
				1: "EXTENDED",
				2: "DIFFUSE",
				3: "DIFFUSE-LARGE",
				4: "ARTEFACT"
			}
			
		else:
			id2target= {
				0: 0, # background
				1: 1, # radio-galaxy
				2: 2, # extended
				3: 3, # diffuse
				4: 4, # diffuse-large
				5: 5, # artefact
			}
		
			id2label= {
				0: "BACKGROUND",
				1: "RADIO-GALAXY",
				2: "EXTENDED",
				3: "DIFFUSE",
				4: "DIFFUSE-LARGE",
				5: "ARTEFACT"
			}
			
	if schema=="morph_tags_B1":
		if skip_first_class:
			id2target= {
				0: -1, # NONE
				1: 0, # EXTENDED
				2: 1, # DIFFUSE
				3: 2, # DIFFUSE-LARGE 
			}
			
			id2label= {
				0: "EXTENDED",
				1: "DIFFUSE",
				2: "DIFFUSE-LARGE",
			}
			
		else:
			id2target= {
				0: 0, # NONE
				1: 1, # EXTENDED
				2: 2, # DIFFUSE
				3: 3, # DIFFUSE-LARGE 
			}
		
			id2label= {
				0: "NONE",
				1: "EXTENDED",
				2: "DIFFUSE",
				3: "DIFFUSE-LARGE",
			}	
			
		
	# - Compute reverse dict
	label2id= {v: k for k, v in id2label.items()}
	
	return id2label, label2id, id2target
	
def get_single_label_target_maps(schema="morph_tags"):
	""" Return single-label classifier target maps """
	
	if schema=="morph_class":
		id2target= {
			1: 0, # 1C-1P
			2: 1, # 1C-2P
			3: 2, # 1C-3P
			4: 3, # 2C-2P
			5: 4, # 2C-3P
			6: 5, # 3C-3P
		}
			
		id2label= {
			0: "1C-1P",
			1: "1C-2P",
			2: "1C-3P",
			3: "2C-2P",
			4: "2C-3P",
			5: "3C-3P"
		}
		
	elif schema=="morph_tags":
		id2target= {
			0: 0, # background
			1: 1, # radio-galaxy
			2: 2, # extended
			3: 3, # diffuse
			4: 4, # diffuse-large
			5: 5, # artefact
		}
		
		id2label= {
			0: "BACKGROUND",
			1: "RADIO-GALAXY",
			2: "EXTENDED",
			3: "DIFFUSE",
			4: "DIFFUSE-LARGE",
			5: "ARTEFACT"
		}
		
	elif schema=="binary_qa":
		id2target= {
			0: 0, 
			1: 1
		}
		
		id2label= {
			0: "NO",
			1: "YES"
		}	
		
	elif schema=="anomaly_class":
		id2target= {
			0: 0, 
			1: 1,
			2: 2
		}
		
		id2label= {
			0: "ORDINARY",
			1: "COMPLEX",
			2: "PECULIAR"
		}	
		
	# - Compute reverse dict
	label2id= {v: k for k, v in id2label.items()}
	
	return id2label, label2id, id2target

######################################
###      DATASET BASE CLASS
######################################
class AstroImageDataset(Dataset):
	""" Dataset to load astro images in FITS format """
	
	def __init__(self, 
		filename, 
		transform=None,
		load_as_gray=False,
		apply_zscale=False,
		zscale_contrast=0.25,
		resize=False,
		resize_size=224,
		verbose=False
	):
		self.filename= filename
		self.datalist= read_datalist(filename)
		self.transform = transform
		self.load_as_gray= load_as_gray
		self.apply_zscale= apply_zscale
		self.zscale_contrast= zscale_contrast
		self.resize= resize
		self.resize_size= resize_size
		self.verbose= verbose
		
		self.pil2tensor = T.Compose([T.PILToTensor()])
		
	def load_pil_image(self, idx):
		""" Load image as PIL """	
		
		# - Get image path
		item= self.datalist[idx]
		image_path= item["filepaths"][0]
		image_ext= os.path.splitext(image_path)[1]
		
		# - Read image (FITS/natural image supported) and then convert to PIL either as 1D or 3-chan image, normalized to [0,1]
		if self.load_as_gray:
			img= load_img_as_pil_float(
				image_path, 
				resize=self.resize, resize_size=self.resize_size, 
				apply_zscale=self.apply_zscale, contrast=self.zscale_contrast, 
				set_nans_to_min=False, 
				verbose=False
			)
		else:
			img= load_img_as_pil_rgb(
				image_path, 
				resize=self.resize, resize_size=self.resize_size, 
				apply_zscale=self.apply_zscale, contrast=self.zscale_contrast, 
				set_nans_to_min=False,
				verbose=False
			)
			
		# - Check for None
		if img is None:
			return None
			
		# - Convert PIL image to tensor if needed
		#if isinstance(img, PIL.Image.Image):
		#	img = self.pil2tensor(img).float()

		# - Replace NaN or Inf with zeros
		#img[~torch.isfinite(img)] = 0

		# - Apply transforms
		#if self.transform:
		#	img = self.transform(img)
		
		return img

		
	def load_npy_image(self, idx):
		""" Load image as numpy """	
		
		# - Get image path
		item= self.datalist[idx]
		image_path= item["filepaths"][0]
		image_ext= os.path.splitext(image_path)[1]
		
		# - Read image (FITS/natural image supported) and then convert to numpy either as 1D or 3-chan image, normalized to [0,1]
		if self.load_as_gray:
			img= load_img_as_npy_float(
				image_path, 
				add_chan_axis=True,
				add_batch_axis=False,
				resize=self.resize, resize_size=self.resize_size, 
				apply_zscale=self.apply_zscale, contrast=self.zscale_contrast, 
				set_nans_to_min=False, 
				verbose=self.verbose
			)
		else:
			img= load_img_as_npy_rgb_float(
				image_path, 
				add_chan_axis=True,
				add_batch_axis=False,
				resize=self.resize, resize_size=self.resize_size, 
				apply_zscale=self.apply_zscale, contrast=self.zscale_contrast, 
				set_nans_to_min=False, 
				verbose=self.verbose
			)
			
		# - Check for None
		if img is None:
			return None
			
		# - Replace NaN or Inf with zeros
		img[~np.isfinite(img)] = 0
		
		#if verbose:
		#	print("npy img")
		#	print(img.dtype)
		#	print(img.shape)
		#	print(img.min())
		#	print(img.max())
			
		return img
				
				
	def load_tensor(self, idx):
		""" Load tensor """			
		
		# - Load image as npy
		img= self.load_npy_image(idx)
		
		# - Check for None
		if img is None:
			return None
		
		# - Convert numpy image to tensor	
		img = torch.from_numpy(img.transpose((2, 0, 1))).contiguous()
		
		# - Replace NaN or Inf with zeros
		img[~torch.isfinite(img)] = 0
		
		#if verbose:
		#	print("tensor img")
		#	print(img.dtype)
		#	print(img.shape)
		#	print(img.min())
		#	print(img.max())
		
		# - Apply transforms
		if self.transform:
			img = self.transform(img)
		
		return img
		
	def load_image_info(self, idx):
		""" Load image metadata """
		return self.datalist[idx]
		
	def __len__(self):
		return len(self.datalist)
			
	def get_sample_size(self):
		return len(self.datalist)
		
		
################################################
###      DATASET (MULTI-LABEL CLASSIFICATION
################################################
class MultiLabelDataset(AstroImageDataset):
	""" Dataset to load astro images in FITS format for multi-label multi-class classification """
	
	def __init__(self, 
		filename, 
		transform=None, 
		load_as_gray=False,
		apply_zscale=False,
		zscale_contrast=0.25,
		resize=False,
		resize_size=224,
		nclasses=None,
		id2target=None,
		#target2label=None,
		#label_schema="radioimg_morph_tags",
		skip_first_class=False,
		verbose=False
	):
		super().__init__(
			filename, 
			transform,
			load_as_gray,
			apply_zscale, zscale_contrast,
			resize, resize_size,
			verbose
		)
		self.skip_first_class= skip_first_class
		#self.label_schema= label_schema
		
		#if nclasses is None or id2target is None or target2label is None:
		#	__set_default_label_schema(label_schema)
		#else:
		#	self.nclasses= nclasses
		#	self.id2target= id2target
		#	self.target2label= target2label
		
		self.nclasses= nclasses
		self.id2target= id2target
		self.mlb = MultiLabelBinarizer(classes=np.arange(0, self.nclasses))

	#def __set_default_label_schema(self, schema):
	#	""" Set the default label schema """
		
	#	if schema=="radioimg_morph_tags":
	#		self.nclasses= 6
	#		self.id2target= {
	#			0: 0, # background
	#			1: 1, # radio-galaxy
	#			2: 2, # extended
	#			3: 3, # diffuse
	#			4: 4, # diffuse-large
	#			5: 5, # artefact
	#		}
	#		self.target2label= {
	#			0: "BACKGROUND",
	#			1: "RADIO-GALAXY",
	#			2: "EXTENDED",
	#			3: "DIFFUSE",
	#			4: "DIFFUSE-LARGE",
	#			5: "ARTEFACT"
	#		}
	
	def __getitem__(self, idx):
		""" Iterator providing training data (pixel_values + labels) """
		
		# - Load image at index as tensor 
		image_tensor= self.load_tensor(idx)
		
		# - Get class ids
		ids= self.datalist[idx]['id']
		class_ids= []
		for id in ids:
			class_id= self.id2target[id]
			if class_id!=-1:
				class_ids.append(class_id)
		
		class_ids.sort()
		
		# - Get class id (hot encoding)
		class_ids_hotenc= self.mlb.fit_transform([class_ids])
		#if self.skip_first_class:
		#	class_ids_hotenc= class_ids_hotenc[:, 1:self.nclasses]
		
		class_ids_hotenc = [j for sub in class_ids_hotenc for j in sub]
		class_ids_hotenc= torch.from_numpy(np.array(class_ids_hotenc).astype(np.float32))
		
		#print("ids")
		#print(ids)
		#print("class_ids")
		#print(class_ids)
		#print("class_ids_hotenc")
		#print(class_ids_hotenc)

		return image_tensor, class_ids_hotenc		
		
################################################
###      DATASET (SINGLE-LABEL CLASSIFICATION
################################################
class SingleLabelDataset(AstroImageDataset):
	""" Dataset to load astro images in FITS format for single-label multi-class classification """
	
	def __init__(self, 
			filename, 
			transform=None, 
			load_as_gray=False,
			apply_zscale=False,
			zscale_contrast=0.25,
			resize=False,
			resize_size=224,
			nclasses=None,
			id2target=None,
			#target2label=None,
			#label_schema="radioimg_morph_tags"
			verbose=False
		):
		super().__init__(
			filename, 
			transform,
			load_as_gray,
			apply_zscale, zscale_contrast,
			resize, resize_size,
			verbose
		)	
		
		#self.label_schema= label_schema
		
		#if nclasses is None or id2target is None or target2label is None:
		#	__set_default_label_schema(label_schema)
		#else:
		#	self.nclasses= nclasses
		#	self.id2target= id2target
		#	self.target2label= target2label
		
		self.nclasses= nclasses
		self.id2target= id2target
		self.mlb = MultiLabelBinarizer(classes=np.arange(0, self.nclasses))

	#def __set_default_label_schema(self, schema):
	#	""" Set the default label schema """
		
	#	if schema=="radioimg_morph_tags":
	#		self.nclasses= 6
	#		self.id2target= {
	#			1: 0, # 1C-1P
	#			2: 1, # 1C-2P
	#			3: 2, # 1C-3P
	#			4: 3, # 2C-2P
	#			5: 4, # 2C-3P
	#			6: 5, # 3C-3P
	#		}
	#		self.target2label= {
	#			0: "1C-1P",
	#			1: "1C-2P",
	#			2: "1C-3P",
	#			3: "2C-2P",
	#			4: "2C-3P",
	#			5: "3C-3P"
	#		}
		
		
	def __getitem__(self, idx):
		""" Iterator providing training data (pixel_values + labels) """
		
		# - Load image at index as tensor 
		image_tensor= self.load_tensor(idx)
		
		# - Get class ids
		class_id= self.datalist[idx]['id']
		target_id= self.id2target[class_id]
		target_ids= [target_id]
		
		# - Get class id (hot encoding)
		target_ids_hotenc= self.mlb.fit_transform([target_ids])	
		target_ids_hotenc = [j for sub in target_ids_hotenc for j in sub]
		target_ids_hotenc= torch.from_numpy(np.array(target_ids_hotenc).astype(np.float32))
		
		return image_tensor, target_ids_hotenc		
		
		
################################################
###      PRETRAIN DATASET
################################################
class PreTrainDataset(AstroImageDataset):
	""" Dataset to load astro images in FITS format for pretraining """
	
	def __init__(self, 
			filename, 
			transform=None, 
			load_as_gray=False,
			apply_zscale=False,
			zscale_contrast=0.25,
			resize=False,
			resize_size=224,
			verbose=False
		):
		super().__init__(
			filename, 
			transform,
			load_as_gray,
			apply_zscale, zscale_contrast,
			resize, resize_size,
			verbose
		)	
		
	def __getitem__(self, idx):
		""" Iterator providing training data (pixel_values + labels) """
		
		# - Load image at index as tensor 
		image_tensor= self.load_tensor(idx)
		
		# - Get class ids (dummy)
		#class_id= self.datalist[idx]['id']
		
		#return image_tensor
		return {"image": image_tensor}
				
def PreTrainDatasetGenerator(dataset: PreTrainDataset):
	""" Generator to convert PyTorch to HuggingFace datasets """
	for i in range(len(dataset)):
		yield dataset[i]
		
