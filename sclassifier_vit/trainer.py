#!/usr/bin/env python

from __future__ import print_function

##################################################
###          MODULE IMPORT
##################################################
# - STANDARD MODULES
import sys
import os
import random
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import gc
from contextlib import nullcontext

# - TORCH
import torch
from torch import nn
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Sampler

# - TRANSFORMERS
import transformers
from transformers import pipeline
from transformers import AutoProcessor, AutoModel
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.data.data_collator import DataCollator
from transformers.training_args import TrainingArguments
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import EvalPrediction
from transformers.trainer_callback import TrainerCallback
from transformers import EvalPrediction    
import evaluate

# - SCLASSIFIER-VIT
from sclassifier_vit.utils import *
from sclassifier_vit import logger

##########################################
##    FOCAL LOSS
##########################################
class FocalLossMultiClass(nn.Module):
	"""
		Multi-class focal loss with logits.
			- alpha: Tensor [C] or float or None (class weighting)
			- gamma: focusing parameter
	"""

	def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
		super().__init__()
		self.gamma = gamma
		self.reduction = reduction
		#self.alpha = alpha  # None | float | Tensor[C]
		self.register_buffer("alpha", alpha if alpha is not None else None)  # stays on the right device 

	def forward(self, logits, targets):
		# - logits: [B, C], targets: [B] int64
		log_probs = F.log_softmax(logits, dim=1)              # [B, C]
		probs = torch.exp(log_probs)                          # [B, C]
		
		# - pick the prob/log_prob of the target class
		pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)       # [B]
		log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B]
		focal_term = (1.0 - pt).clamp_min(1e-8).pow(self.gamma)      # [B]

		if self.alpha is None:
			alpha_t = 1.0
		elif isinstance(self.alpha, float):
			alpha_t = self.alpha
		else:
			# alpha is tensor [C]
			alpha_t = self.alpha.to(logits.device).gather(0, targets)

		loss = -alpha_t * focal_term * log_pt  # [B]
		if self.reduction == "mean":
			return loss.mean()
		elif self.reduction == "sum":
			return loss.sum()
		else:
			return loss


class FocalLossMultiLabel(nn.Module):
	"""
		Multi-label focal loss with logits (BCE variant).
		pos_weight acts like class-wise alpha for positives.
	"""
    
	def __init__(self, gamma=2.0, pos_weight=None, reduction="mean"):
		super().__init__()
		self.gamma = gamma
		self.pos_weight = pos_weight  # Tensor [C] or None
		self.reduction = reduction

	def forward(self, logits, targets):
		# logits: [B, C], targets: [B, C] in {0,1}
		# stable BCE terms
		bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=self.pos_weight)
		# pt for each class
		p = torch.sigmoid(logits)
		pt = torch.where(targets == 1, p, 1 - p)
		focal_term = (1 - pt).clamp_min(1e-8).pow(self.gamma)

		loss = focal_term * bce
		if self.reduction == "mean":
			return loss.mean()
		elif self.reduction == "sum":
			return loss.sum()
		else:
			return loss
			
			
##########################################
##    CUSTOM TRAINER
##########################################			
class CustomTrainer(transformers.Trainer):
	def __init__(
  	self,
  	*args,
  	multilabel=False,
  	class_weights=None,          # torch.tensor [C] or None
		loss_type="ce",              # "ce" or "focal"
		focal_gamma=2.0,
		focal_alpha=None,            # None | float | tensor[C] (multiclass)
		binary_pos_weights=None,
		logitout_size=4,
  	**kwargs
	):
		super().__init__(*args, **kwargs)
		
		# - Set class vars
		self.multilabel= multilabel
		self.class_weights = class_weights
		self.loss_type = loss_type
		self.focal_gamma = focal_gamma
		self.focal_alpha = focal_alpha
		self.binary_pos_weights= binary_pos_weights
		self.logitout_size= logitout_size
		self.is_binary_single_logit = (
			self.logitout_size==1
		)
		self.dev = next(self.model.parameters()).device
		
		# - Set loss fcn
		self._set_loss_fcn()
		
	def _set_loss_fcn(self):
		""" Set loss function """
		
		if self.multilabel:
			return self._set_multilabel_class_loss_fcn()
		else:
			return self._set_singlelabel_class_loss_fcn()
		
	def _set_singlelabel_class_loss_fcn(self):
		""" Set loss function for single-label classification """
			
		if self.loss_type == "ce":
			if self.is_binary_single_logit:
				# BCE for single-logit binary
				pos_w = (self.binary_pos_weights.to(self.dev) if self.binary_pos_weights is not None else None)
				self.loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
			else:
				w = self.class_weights.to(self.dev) if self.class_weights is not None else None
				self.loss_fct = torch.nn.CrossEntropyLoss(weight=w)	
				
		elif self.loss_type == "focal":
			if self.is_binary_single_logit:
				# Use BCE-style focal via multilabel focal (C=1)
				pos_w = (self.binary_pos_weights.to(self.dev) if self.binary_pos_weights is not None else None)
				self.loss_fct = FocalLossMultiLabel(
					gamma=self.focal_gamma,
					pos_weight=pos_w,      # class tilt for positives
					reduction="mean",
				)
			else:
				alpha = self.focal_alpha
				if isinstance(alpha, torch.Tensor):
					alpha = alpha.to(self.dev)
				self.loss_fct = FocalLossMultiClass(alpha=alpha, gamma=self.focal_gamma, reduction="mean")
					
		else:
			raise ValueError(f"Unknown loss_type: {self.loss_type}")
				
			
	def _set_multilabel_class_loss_fcn(self):
		""" Set loss function for multi-label classification """
		
		if self.loss_type == "ce":
			pos_w = self.class_weights.to(self.dev) if self.class_weights is not None else None
			self.loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
				
		elif self.loss_type == "focal":
			pos_w = self.class_weights.to(self.dev) if self.class_weights is not None else None
			self.loss_fct = FocalLossMultiLabel(gamma=self.focal_gamma, pos_weight=pos_w, reduction="mean")
			
		else:
			raise ValueError(f"Unknown/unsupported loss_type for multilabel classification: {self.loss_type}")
		
	def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
		""" Override trainer compute_loss function """
		
		# - Retrieve pixel values
		pixel_values= inputs.get("pixel_values")
		if torch.is_tensor(pixel_values):
			if torch.isnan(pixel_values).any():
				print("⚠️ NaN values detected in batch features tensor!")
			if torch.isinf(pixel_values).any():
				print("⚠️ Inf values detected in batch features tensor!")
		
		# - Retrieve labels
		labels= inputs.get("labels")
		if torch.isnan(labels).any() or torch.isinf(labels).any():
			print("⚠️ NaN values detected in batch label tensor!")
		
		# - Retrieve logits
		outputs = model(pixel_values)
		logits = outputs.logits
		if torch.isnan(logits).any() or torch.isinf(logits).any():
			print("⚠️ NaN values detected in logits tensor!")	
		
		# - Shape fix for single-logit binary + BCE/focal ----
		if self.is_binary_single_logit and self.loss_type in ("ce", "focal"):
			# logits: (B,1), labels: (B,) -> (B,1)
			labels = labels.float().view(-1, 1)

		if self.multilabel or (self.is_binary_single_logit and self.loss_type in ("ce","focal")):
			# BCE-style losses expect float targets
			labels = labels.float()
		
		# - Compute loss
		loss = self.loss_fct(logits, labels)
			
		return (loss, outputs) if return_outputs else loss
	
	
	def log_metrics(self, split, metrics):
		""" Override log metrics to enable print of dict/list metrics """
		
		if self.multilabel:
			return self._log_metrics_multilabel_class(split, metrics)
		else:
			return self._log_metrics_singlelabel_class(split, metrics)
	
	def _log_metrics_singlelabel_class(self, split, metrics):
		""" Override log metrics to enable print of dict/list metrics for single-label classification """
		
		if not self.is_world_process_zero():
			return
			
		print(f"***** {split} metrics *****")
		try:
			metrics_formatted = self.metrics_format(metrics)
		except Exception as e:
			logger.warning(f"Syntax error when formatting the metrics (err={str(e)}), retrying with another method ...")
			metrics_formatted = transformers.Trainer.metrics_format(metrics)
		
		# - Find class names
		class_names= []
		for key in metrics_formatted:
			if 'class_names' in key:
				class_names= metrics_formatted[key]		
		
		max_class_name_length= max([len(x) for x in class_names], default=0)
		print("max_class_name_length=%d" % (max_class_name_length))
		
		metric_names_for_format= []
		metric_values_for_format= []
		for x in metrics_formatted.keys():
			if "class_names" in x:
				continue
			elif "accuracy_class" in x:
				continue
			elif "class_report" in x:
				continue
			else:
				metric_names_for_format.append(x)
				metric_values_for_format.append(metrics_formatted[x])
			
		#k_width = max(len(str(x)) for x in metrics_formatted.keys())
		#v_width = max(len(str(x)) for x in metrics_formatted.values())
		k_width = max(len(str(x)) for x in metric_names_for_format) + max_class_name_length
		v_width = max(len(str(x)) for x in metric_values_for_format)
		
		
		print("log_metrics-->class_names")
		print(class_names)
		
		# - Print formatted metrics
		for key in sorted(metrics_formatted.keys()):
			if "class_names" in key:
				continue
			elif "class_report" in key:
				# - Print class #instances
				for class_name in class_names:
					ninstances= metrics_formatted[key][class_name]['support']
					metric_name= key.split("_")[0] + '_nsamples_class_' + class_name
					metric_val= ninstances
					print(f"  {metric_name: <{k_width}} = {metric_val:>{v_width}}")
				
				# - Print class precision
				for class_name in class_names:
					precision= metrics_formatted[key][class_name]['precision']
					metric_name= key.split("_")[0] + '_precision_class_' + class_name
					metric_val= precision
					print(f"  {metric_name: <{k_width}} = {metric_val:>{v_width}}")
				
				# - Print class recall
				for class_name in class_names:
					recall= metrics_formatted[key][class_name]['recall']
					metric_name= key.split("_")[0] + '_recall_class_' + class_name
					metric_val= recall
					print(f"  {metric_name: <{k_width}} = {metric_val:>{v_width}}")
				
				# - Print class F1score
				for class_name in class_names:
					f1score= metrics_formatted[key][class_name]['f1-score']
					metric_name= key.split("_")[0] + '_f1score_class_' + class_name
					metric_val= f1score
					print(f"  {metric_name: <{k_width}} = {metric_val:>{v_width}}")
			
			elif "confusion_matrix" in key:
				print(f"  {metric_name: <{k_width}} ")
				print(metrics_formatted[key])
			
			elif "confusion_matrix_norm" in key:
				print(f"  {metric_name: <{k_width}} ")
				print(metrics_formatted[key])
				
			else:
				metrics_str= str(metrics_formatted[key])
				print(f"  {key: <{k_width}} = {metrics_str:>{v_width}}")
		
		
	def _log_metrics_multilabel_class(self, split, metrics):
		""" Override log metrics to enable print of dict/list metrics for multi-label classification """
		
		if not self.is_world_process_zero():
			return
			
		print(f"***** {split} metrics *****")
		try:
			metrics_formatted = self.metrics_format(metrics)
		except Exception as e:
			logger.warning(f"Syntax error when formatting the metrics (err={str(e)}), retrying with another method ...")
			metrics_formatted = transformers.Trainer.metrics_format(metrics)
		
		# - Find class names
		class_names= []
		for key in metrics_formatted:
			if 'class_names' in key:
				class_names= metrics_formatted[key]		
		
		max_class_name_length= max([len(x) for x in class_names], default=0)
		print("max_class_name_length=%d" % (max_class_name_length))
		
		metric_names_for_format= []
		metric_values_for_format= []
		for x in metrics_formatted.keys():
			if "class_names" in x:
				continue
			elif "accuracy_class" in x:
				continue
			elif "class_report" in x:
				continue
			else:
				metric_names_for_format.append(x)
				metric_values_for_format.append(metrics_formatted[x])
			
		#k_width = max(len(str(x)) for x in metrics_formatted.keys())
		#v_width = max(len(str(x)) for x in metrics_formatted.values())
		k_width = max(len(str(x)) for x in metric_names_for_format) + max_class_name_length
		v_width = max(len(str(x)) for x in metric_values_for_format)
		
		print("log_metrics-->class_names")
		print(class_names)
		
		# - Print formatted metrics
		for key in sorted(metrics_formatted.keys()):
			if "class_names" in key:
				continue
			elif "accuracy_class" in key:
				for i in range(len(metrics_formatted[key])):
					metric_name= key + '_' + class_names[i]
					metric_val= metrics_formatted[key][i]
					print(f"  {metric_name: <{k_width}} = {metric_val:>{v_width}}")
			elif "class_report" in key:
				# - Print class #instances
				for class_name in class_names:
					ninstances= metrics_formatted[key][class_name]['support']
					metric_name= key.split("_")[0] + '_nsamples_class_' + class_name
					metric_val= ninstances
					print(f"  {metric_name: <{k_width}} = {metric_val:>{v_width}}")
				
				# - Print class precision
				for class_name in class_names:
					precision= metrics_formatted[key][class_name]['precision']
					metric_name= key.split("_")[0] + '_precision_class_' + class_name
					metric_val= precision
					print(f"  {metric_name: <{k_width}} = {metric_val:>{v_width}}")
				
				# - Print class recall
				for class_name in class_names:
					recall= metrics_formatted[key][class_name]['recall']
					metric_name= key.split("_")[0] + '_recall_class_' + class_name
					metric_val= recall
					print(f"  {metric_name: <{k_width}} = {metric_val:>{v_width}}")
				
				# - Print class F1score
				for class_name in class_names:
					f1score= metrics_formatted[key][class_name]['f1-score']
					metric_name= key.split("_")[0] + '_f1score_class_' + class_name
					metric_val= f1score
					print(f"  {metric_name: <{k_width}} = {metric_val:>{v_width}}")
				
			else:
				metrics_str= str(metrics_formatted[key])
				print(f"  {key: <{k_width}} = {metrics_str:>{v_width}}")
		
##########################################
##    MULTI-LABEL CLASSIFIER TRAINER
##########################################
class MultiLabelClassTrainer(transformers.Trainer):
	def __init__(
  	self,
  	*args,
  	num_labels,
  	class_weights=None,          # torch.tensor [C] or None
		loss_type="ce",              # "ce" or "focal"
		focal_gamma=2.0,
		focal_alpha=None,            # None | float | tensor[C] (multiclass)
		binary_pos_weights=None,
		logitout_size=4,
  	**kwargs
	):
		super().__init__(*args, **kwargs)
		
	#def __init__(
  #	self,
  #	num_labels,
	#	model: Union[PreTrainedModel, nn.Module] = None,
	#	args: TrainingArguments = None,
	#	data_collator: Optional[DataCollator] = None,
	#	train_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None,
	#	eval_dataset: Optional[Union[Dataset, Dict[str, Dataset], "datasets.Dataset"]] = None,
	#	tokenizer: Optional[PreTrainedTokenizerBase] = None,
	#	model_init: Optional[Callable[[], PreTrainedModel]] = None,
	#	compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
	#	callbacks: Optional[List[TrainerCallback]] = None,
	#	optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
	#	preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
	#):
	#	super().__init__(
	#		model=model, 
	#		args=args,
	#		data_collator=data_collator,
	#		train_dataset=train_dataset,
	#		eval_dataset=eval_dataset,
	#		tokenizer=tokenizer,
	#		model_init=model_init,
	#		compute_metrics=compute_metrics,
	#		callbacks=callbacks,
	#		optimizers=optimizers,
	#		preprocess_logits_for_metrics=preprocess_logits_for_metrics
	#	)
	
		# - Set class vars
		self.num_labels= num_labels
		self.class_weights = class_weights
		self.loss_type = loss_type
		self.focal_gamma = focal_gamma
		self.focal_alpha = focal_alpha
		self.binary_pos_weights= binary_pos_weights
		self.logitout_size= logitout_size
		self.is_binary_single_logit = (
			self.logitout_size==1
		)
		
		dev = next(self.model.parameters()).device
		
		# - Create loss fcn
		if self.loss_type == "ce":
			pos_w = self.class_weights.to(dev) if self.class_weights is not None else None
			self.loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
				
		elif self.loss_type == "focal":
			pos_w = self.class_weights.to(dev) if self.class_weights is not None else None
			self.loss_fct = FocalLossMultiLabel(gamma=self.focal_gamma, pos_weight=pos_w, reduction="mean")
			
		else:
			raise ValueError(f"Unknown/unsupported loss_type for multilabel classification: {self.loss_type}")
	
	
	def log_metrics(self, split, metrics):
		""" Override log metrics to enable print of dict/list metrics """
		
		if not self.is_world_process_zero():
			return
			
		print(f"***** {split} metrics *****")
		try:
			metrics_formatted = self.metrics_format(metrics)
		except Exception as e:
			logger.warning(f"Syntax error when formatting the metrics (err={str(e)}), retrying with another method ...")
			metrics_formatted = transformers.Trainer.metrics_format(metrics)
		
		# - Find class names
		class_names= []
		for key in metrics_formatted:
			if 'class_names' in key:
				class_names= metrics_formatted[key]		
		
		max_class_name_length= max([len(x) for x in class_names], default=0)
		print("max_class_name_length=%d" % (max_class_name_length))
		
		metric_names_for_format= []
		metric_values_for_format= []
		for x in metrics_formatted.keys():
			if "class_names" in x:
				continue
			elif "accuracy_class" in x:
				continue
			elif "class_report" in x:
				continue
			else:
				metric_names_for_format.append(x)
				metric_values_for_format.append(metrics_formatted[x])
			
		#k_width = max(len(str(x)) for x in metrics_formatted.keys())
		#v_width = max(len(str(x)) for x in metrics_formatted.values())
		k_width = max(len(str(x)) for x in metric_names_for_format) + max_class_name_length
		v_width = max(len(str(x)) for x in metric_values_for_format)
		
		print("log_metrics-->class_names")
		print(class_names)
		
		# - Print formatted metrics
		for key in sorted(metrics_formatted.keys()):
			if "class_names" in key:
				continue
			elif "accuracy_class" in key:
				for i in range(len(metrics_formatted[key])):
					metric_name= key + '_' + class_names[i]
					metric_val= metrics_formatted[key][i]
					print(f"  {metric_name: <{k_width}} = {metric_val:>{v_width}}")
			elif "class_report" in key:
				# - Print class #instances
				for class_name in class_names:
					ninstances= metrics_formatted[key][class_name]['support']
					metric_name= key.split("_")[0] + '_nsamples_class_' + class_name
					metric_val= ninstances
					print(f"  {metric_name: <{k_width}} = {metric_val:>{v_width}}")
				
				# - Print class precision
				for class_name in class_names:
					precision= metrics_formatted[key][class_name]['precision']
					metric_name= key.split("_")[0] + '_precision_class_' + class_name
					metric_val= precision
					print(f"  {metric_name: <{k_width}} = {metric_val:>{v_width}}")
				
				# - Print class recall
				for class_name in class_names:
					recall= metrics_formatted[key][class_name]['recall']
					metric_name= key.split("_")[0] + '_recall_class_' + class_name
					metric_val= recall
					print(f"  {metric_name: <{k_width}} = {metric_val:>{v_width}}")
				
				# - Print class F1score
				for class_name in class_names:
					f1score= metrics_formatted[key][class_name]['f1-score']
					metric_name= key.split("_")[0] + '_f1score_class_' + class_name
					metric_val= f1score
					print(f"  {metric_name: <{k_width}} = {metric_val:>{v_width}}")
				
			else:
				metrics_str= str(metrics_formatted[key])
				print(f"  {key: <{k_width}} = {metrics_str:>{v_width}}")

	def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
		""" Override trainer compute_loss function """
		
		pixel_values= inputs.get("pixel_values")
		labels= inputs.get("labels")
		
		outputs = model(pixel_values)
		logits = outputs.logits
		
		loss_fct = torch.nn.BCEWithLogitsLoss() 
		loss= loss_fct(logits.view(-1,self.num_labels), labels.type_as(logits).view(-1,self.num_labels)) #convert labels to float for calculation
		
		return (loss, outputs) if return_outputs else loss
		
	
##########################################
##    SINGLE-LABEL CLASSIFIER TRAINER
##########################################
class SingleLabelClassTrainer(transformers.Trainer):
	def __init__(
  	self,
  	*args,
  	num_labels,
  	class_weights=None,          # torch.tensor [C] or None
		loss_type="ce",              # "ce" or "focal"
		focal_gamma=2.0,
		focal_alpha=None,            # None | float | tensor[C] (multiclass)
		binary_pos_weights=None,
		logitout_size=4,
  	**kwargs
	):
		super().__init__(*args, **kwargs)
 
 		# - Set class vars
		self.num_labels= num_labels
		self.class_weights = class_weights
		self.loss_type = loss_type
		self.focal_gamma = focal_gamma
		self.focal_alpha = focal_alpha
		self.binary_pos_weights= binary_pos_weights
		self.logitout_size= logitout_size
		self.is_binary_single_logit = (
			self.logitout_size==1
		)
		
		dev = next(self.model.parameters()).device
		
		# - Create loss
		if self.loss_type == "ce":
			if self.is_binary_single_logit:
				# BCE for single-logit binary
				pos_w = (self.binary_pos_weights.to(dev) if self.binary_pos_weights is not None else None)
				self.loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
			else:
				w = self.class_weights.to(dev) if self.class_weights is not None else None
				self.loss_fct = torch.nn.CrossEntropyLoss(weight=w)	
				
		elif self.loss_type == "focal":
			if self.is_binary_single_logit:
				# Use BCE-style focal via multilabel focal (C=1)
				pos_w = (self.binary_pos_weights.to(dev) if self.binary_pos_weights is not None else None)
				self.loss_fct = FocalLossMultiLabel(
					gamma=self.focal_gamma,
					pos_weight=pos_w,      # class tilt for positives
					reduction="mean",
				)
			else:
				alpha = self.focal_alpha
				if isinstance(alpha, torch.Tensor):
					alpha = alpha.to(dev)
				self.loss_fct = FocalLossMultiClass(alpha=alpha, gamma=self.focal_gamma, reduction="mean")
					
		else:
			raise ValueError(f"Unknown loss_type: {self.loss_type}")
		
	
	def log_metrics(self, split, metrics):
		""" Override log metrics to enable print of dict/list metrics """
		
		if not self.is_world_process_zero():
			return
			
		
		print(f"***** {split} metrics *****")
		try:
			metrics_formatted = self.metrics_format(metrics)
		except Exception as e:
			logger.warning(f"Syntax error when formatting the metrics (err={str(e)}), retrying with another method ...")
			metrics_formatted = transformers.Trainer.metrics_format(metrics)
		
		# - Find class names
		class_names= []
		for key in metrics_formatted:
			if 'class_names' in key:
				class_names= metrics_formatted[key]		
		
		max_class_name_length= max([len(x) for x in class_names], default=0)
		print("max_class_name_length=%d" % (max_class_name_length))
		
		metric_names_for_format= []
		metric_values_for_format= []
		for x in metrics_formatted.keys():
			if "class_names" in x:
				continue
			elif "accuracy_class" in x:
				continue
			elif "class_report" in x:
				continue
			else:
				metric_names_for_format.append(x)
				metric_values_for_format.append(metrics_formatted[x])
			
		#k_width = max(len(str(x)) for x in metrics_formatted.keys())
		#v_width = max(len(str(x)) for x in metrics_formatted.values())
		k_width = max(len(str(x)) for x in metric_names_for_format) + max_class_name_length
		v_width = max(len(str(x)) for x in metric_values_for_format)
		
		
		print("log_metrics-->class_names")
		print(class_names)
		
		# - Print formatted metrics
		for key in sorted(metrics_formatted.keys()):
			if "class_names" in key:
				continue
			elif "class_report" in key:
				# - Print class #instances
				for class_name in class_names:
					ninstances= metrics_formatted[key][class_name]['support']
					metric_name= key.split("_")[0] + '_nsamples_class_' + class_name
					metric_val= ninstances
					print(f"  {metric_name: <{k_width}} = {metric_val:>{v_width}}")
				
				# - Print class precision
				for class_name in class_names:
					precision= metrics_formatted[key][class_name]['precision']
					metric_name= key.split("_")[0] + '_precision_class_' + class_name
					metric_val= precision
					print(f"  {metric_name: <{k_width}} = {metric_val:>{v_width}}")
				
				# - Print class recall
				for class_name in class_names:
					recall= metrics_formatted[key][class_name]['recall']
					metric_name= key.split("_")[0] + '_recall_class_' + class_name
					metric_val= recall
					print(f"  {metric_name: <{k_width}} = {metric_val:>{v_width}}")
				
				# - Print class F1score
				for class_name in class_names:
					f1score= metrics_formatted[key][class_name]['f1-score']
					metric_name= key.split("_")[0] + '_f1score_class_' + class_name
					metric_val= f1score
					print(f"  {metric_name: <{k_width}} = {metric_val:>{v_width}}")
			
			elif "confusion_matrix" in key:
				print(f"  {metric_name: <{k_width}} ")
				print(metrics_formatted[key])
			
			elif "confusion_matrix_norm" in key:
				print(f"  {metric_name: <{k_width}} ")
				print(metrics_formatted[key])
				
			else:
				metrics_str= str(metrics_formatted[key])
				print(f"  {key: <{k_width}} = {metrics_str:>{v_width}}")

	def compute_loss_old(self, model, inputs, return_outputs=False, num_items_in_batch=None):
		""" Override trainer compute_loss function """
		
		pixel_values= inputs.get("pixel_values")
		labels= inputs.get("labels")
		
		outputs = model(pixel_values)
		logits = outputs.logits
		
		loss_fct = torch.nn.CrossEntropyLoss()
		loss= loss_fct(logits.view(-1,self.num_labels), labels.type_as(logits).view(-1,self.num_labels)) #convert labels to float for calculation
		
		return (loss, outputs) if return_outputs else loss	
	
	def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
		""" Override trainer compute_loss function """
		
		# - Retrieve pixel values
		pixel_values= inputs.get("pixel_values")
		if torch.is_tensor(pixel_values):
			if torch.isnan(pixel_values).any():
				print("⚠️ NaN values detected in batch features tensor!")
			if torch.isinf(pixel_values).any():
				print("⚠️ Inf values detected in batch features tensor!")
		
		# - Retrieve labels
		labels= inputs.get("labels")
		if torch.isnan(labels).any() or torch.isinf(labels).any():
			print("⚠️ NaN values detected in batch label tensor!")
		
		# - Retrieve logits
		outputs = model(pixel_values)
		logits = outputs.logits
		if torch.isnan(logits).any() or torch.isinf(logits).any():
			print("⚠️ NaN values detected in logits tensor!")	
		
		# - Shape fix for single-logit binary + BCE/focal ----
		if self.is_binary_single_logit and self.loss_type in ("ce", "focal"):
			# logits: (B,1), labels: (B,) -> (B,1)
			labels = labels.float().view(-1, 1)

		if self.is_binary_single_logit and self.loss_type in ("ce","focal"):
			# BCE-style losses expect float targets
			labels = labels.float()
		
		# - Compute loss
		loss = self.loss_fct(logits, labels)
			
		return (loss, outputs) if return_outputs else loss
	
