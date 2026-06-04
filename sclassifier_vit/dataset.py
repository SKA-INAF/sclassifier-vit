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
		# RadioGalaxyZoo (RGZ) dataset
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
		
	elif schema=="rg_morph_binary":
		# MIRABEST dataset
		id2target= {
			1: 0, # FR-I
			2: 1, # FR-II
		}
			
		id2label= {
			0: "FR-I",
			1: "FR-II",
		}
		
	elif schema=="rg_morph":
		# LOTSS DR2 Horton dataset
		id2target= {
			0: 0, # FR-I
			1: 1, # FR-II
			2: 2, # HYBRID
			3: 3, # OTHER
		}
			
		id2label= {
			0: "FR-I",
			1: "FR-II",
			2: "HYBRID",
			3: "OTHER"
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
		self.filename_fullpath= os.path.abspath(self.filename)
		self.filename_base= os.path.basename(self.filename_fullpath)
		self.filename_base_noext= os.path.splitext(self.filename_base)[0]
		self.filename_ext= os.path.splitext(self.filename_base)[1]
		if self.filename_ext==".json":
			self.datalist= read_datalist(filename)
		elif self.filename_ext in {".fits", ".png", ".jpg", ".jpeg"}:
			self.datalist= self._set_datalist_from_image(filename)
		else:
			err= f"Unknow/unsupported input file extension ({self.filename_ext})!"
			logger.error(err)
			raise RuntimeError(err)
			
		self.transform = transform
		self.load_as_gray= load_as_gray
		self.apply_zscale= apply_zscale
		self.zscale_contrast= zscale_contrast
		self.resize= resize
		self.resize_size= resize_size
		self.verbose= verbose
		
		self.pil2tensor = T.Compose([T.PILToTensor()])
		
	def _set_datalist_from_image(self, filename):
		""" Set data list from input image """
		
		filename_fullpath= os.path.abspath(filename)
		filename_base= os.path.basename(filename_fullpath)
		filename_base_noext= os.path.splitext(filename_base)[0]
		
		datalist= [
			{
				"filepaths": [filename],
				"sname": filename_base_noext
			}
		]
		return datalist
		
		
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
		
	def load_target(self, idx, id2target):
		""" Load single-class/single-out target """
			
		# - Get class ids
		id= self.datalist[idx]['id']
		class_id= id2target[id]
		
		return class_id	
		
	def load_targets(self, idx, id2target):
		""" Load single-class/single-out target """
			
		# - Get class ids
		ids= self.datalist[idx]['ids']
		class_ids= [id2target[id] for id in ids]
		return class_ids	
		
	def load_hotenc_targets(self, idx, id2target, mlb):
		""" Load multi-class/multi-out targets """	
		
		# - Get class ids
		class_ids= self.load_targets(idx, id2target)
		
		# - Get class id (hot encoding)
		class_ids_hotenc= [mlb.fit_transform([[id]]) for id in class_ids]
		class_ids_hotenc = [j for sub in class_ids_hotenc for j in sub]
		class_ids_hotenc= torch.from_numpy(np.array(class_ids_hotenc).astype(np.float32))
		
		return class_ids_hotenc	
		
	def load_image_info(self, idx):
		""" Load image metadata """
		return self.datalist[idx]
		
	def __len__(self):
		return len(self.datalist)
			
	def get_sample_size(self):
		return len(self.datalist)
		
	def compute_class_weights(
		self, 
		num_classes, 
		id2target, 
		scheme="balanced", 
		normalize=True,
		binary=False,
		positive_label=1,
		laplace=1.0
	):
		""" Compute class weights from dataset """
    
		# - Collect labels
		ys = []
		for i in range(len(self.datalist)):
			y= self.load_target(i, id2target)
			ys.append(int(y))
		
		counts = np.bincount(ys, minlength=num_classes).astype(float)
		
		print("counts")
		print(counts)

		# --- Binary path: also provide BCE pos_weight + optional per-sample weights
		if binary and num_classes == 2:
			pos_idx = int(positive_label)
			neg_idx = 1 - pos_idx

			# Laplace smoothing to avoid divide-by-zero on rare/empty class
			pos_s = counts[pos_idx] + laplace
			neg_s = counts[neg_idx] + laplace

			# BCEWithLogitsLoss: pos_weight multiplies the positive examples in the loss
			# canonical choice ≈ N_neg / N_pos (do NOT normalize this)
			class_weights = torch.tensor([neg_s / pos_s], dtype=torch.float32) # length-1 tensor: [N_neg/N_pos]
       
		else:
			if scheme == "inverse":
				w = 1.0 / np.maximum(counts, 1.0)
			elif scheme == "inverse_v2":
				w = np.max(counts)/counts
			else:
				# "balanced" like sklearn: n_samples / (n_classes * count_c)
				n = counts.sum()
				w = n / (num_classes * np.maximum(counts, 1.0))

			# optional normalization (keeps average weight ~1)
			print("weights")
			print(w)
			
			if normalize:
				w = w * (num_classes / w.sum())
				print("weights (after norm)")
				print(w)
		
			class_weights = torch.tensor(w, dtype=torch.float32)	
		
		return class_weights
		
	def compute_mild_focal_alpha_from_dataset(
		self,
		num_classes,
		id2target,
		use_flareid= False,
		exponent= 0.5,     # use 0.5 for sqrt inverse-frequency
		cap_ratio= 10.0,   # cap at <= 10× median weight
		device= "cpu",
		dtype= torch.float32,
	):
		"""
			Returns a torch.Tensor of shape [num_classes] to be used as focal alpha.
				- Start from class frequencies (counts / total).
				- Compute inverse-frequency^exponent (e.g., 1/sqrt(freq)).
				- Normalize so mean(alpha)=1 (nice for loss scale).
				- Cap at <= cap_ratio × median(alpha).
		"""

		# - Collect labels
		ys = []
		if use_flareid:
			for i in range(len(self.datalist)):
				y= self.load_flareid(i)
				ys.append(int(y))
		else:
			for i in range(len(self.datalist)):
				y= self.load_target(i, id2target)
				ys.append(int(y))		

		counts = np.bincount(ys, minlength=num_classes).astype(float)
		total = counts.sum()
    
		# - Avoid division by zero and handle missing classes
		eps = 1e-8
		freqs = counts / max(total, 1)
		freqs = np.clip(freqs, eps, 1.0)

		# - inverse-frequency^exponent
		inv = np.power(1.0 / freqs, exponent)

		# - normalize: mean(alpha)=1 (keeps loss scale stable)
		alpha = inv / inv.mean()

		# - cap extremes: <= cap_ratio × median(alpha)
		med = np.median(alpha)
		cap = med * cap_ratio
		alpha = np.minimum(alpha, cap)

		# - (optional) tiny floor to avoid exact zeros after numeric ops
		alpha = np.maximum(alpha, 1e-6)

		# - torch tensor
		alpha_t = torch.tensor(alpha, dtype=dtype, device=device)
		
		return alpha_t, counts	
		
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
			verbose=False,
			return_dict=False
		):
		super().__init__(
			filename, 
			transform,
			load_as_gray,
			apply_zscale, zscale_contrast,
			resize, resize_size,
			verbose
		)	
		self.return_dict= return_dict
			
	def __getitem__(self, idx):
		""" Iterator providing training data (pixel_values + labels) """
		
		# - Load image at index as tensor 
		image_tensor= self.load_tensor(idx)
		
		# - Get class ids (dummy)
		#class_id= self.datalist[idx]['id']
		
		if self.return_dict:
			return {"image": image_tensor}
		else:
			return image_tensor

				
def PreTrainDatasetGenerator(dataset: PreTrainDataset):
	""" Generator to convert PyTorch to HuggingFace datasets """
	for i in range(len(dataset)):
		yield dataset[i]
		
