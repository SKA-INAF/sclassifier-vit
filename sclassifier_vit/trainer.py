#!/usr/bin/env python

from __future__ import print_function

##################################################
###          MODULE IMPORT
##################################################
# - STANDARD MODULES
import sys
import os
import random

# - TORCH
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T

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
##    MULTI-LABEL CLASSIFIER TRAINER
##########################################
class MultiLabelClassTrainer(transformers.Trainer):
	def __init__(
  	self,
  	num_labels,
		model: Union[PreTrainedModel, nn.Module] = None,
		args: TrainingArguments = None,
		data_collator: Optional[DataCollator] = None,
		train_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None,
		eval_dataset: Optional[Union[Dataset, Dict[str, Dataset], "datasets.Dataset"]] = None,
		tokenizer: Optional[PreTrainedTokenizerBase] = None,
		model_init: Optional[Callable[[], PreTrainedModel]] = None,
		compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
		callbacks: Optional[List[TrainerCallback]] = None,
		optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
		preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
	):
		super().__init__(
			model, 
			args,
			data_collator,
			train_dataset,
			eval_dataset,
			tokenizer,
			model_init,
			compute_metrics,
			callbacks,
			optimizers,
			preprocess_logits_for_metrics
		)
		self.num_labels= num_labels
	
	
	def log_metrics(self, split, metrics):
		""" Override log metrics to enable print of dict/list metrics """
		
		if not self.is_world_process_zero():
			return
			
		print(f"***** {split} metrics *****")
		metrics_formatted = self.metrics_format(metrics)
		
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

	def compute_loss(self, model, inputs, return_outputs=False):
		""" Override trainer compute_loss function """
		
		pixel_values= inputs.get("pixel_values")
		labels= inputs.get("labels")
		
		outputs = model(pixel_values)
		logits = outputs.logits
		
		loss_fct = torch.nn.BCEWithLogitsLoss() 
		loss= loss_fct(logits.view(-1,self.num_labels), labels.type_as(logits).view(-1,self.num_labels)) #convert labels to float for calculation
		
		return (loss, outputs) if return_outputs else loss
		
	#def __compute_focal_loss():
	#	""" Compute focal loss """

	#	BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
	#	pt = torch.exp(-BCE_loss) # prevents nans when probability 0
	#	F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
	#	return focal_loss.mean()
	
	
##########################################
##    SINGLE-LABEL CLASSIFIER TRAINER
##########################################
class SingleLabelClassTrainer(transformers.Trainer):
	def __init__(
  	self,
  	num_labels,
		model: Union[PreTrainedModel, nn.Module] = None,
		args: TrainingArguments = None,
		data_collator: Optional[DataCollator] = None,
		train_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None,
		eval_dataset: Optional[Union[Dataset, Dict[str, Dataset], "datasets.Dataset"]] = None,
		tokenizer: Optional[PreTrainedTokenizerBase] = None,
		model_init: Optional[Callable[[], PreTrainedModel]] = None,
		compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
		callbacks: Optional[List[TrainerCallback]] = None,
		optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
		preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
	):
		super().__init__(
			model, 
			args,
			data_collator,
			train_dataset,
			eval_dataset,
			tokenizer,
			model_init,
			compute_metrics,
			callbacks,
			optimizers,
			preprocess_logits_for_metrics
		)
		self.num_labels= num_labels
	
	
	def log_metrics(self, split, metrics):
		""" Override log metrics to enable print of dict/list metrics """
		
		if not self.is_world_process_zero():
			return
			
		
		print(f"***** {split} metrics *****")
		metrics_formatted = self.metrics_format(metrics)
		
		
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

	def compute_loss(self, model, inputs, return_outputs=False):
		""" Override trainer compute_loss function """
		
		pixel_values= inputs.get("pixel_values")
		labels= inputs.get("labels")
		
		outputs = model(pixel_values)
		logits = outputs.logits
		#print("logits")
		#print(logits)
		
		loss_fct = torch.nn.CrossEntropyLoss()
		loss= loss_fct(logits.view(-1,self.num_labels), labels.type_as(logits).view(-1,self.num_labels)) #convert labels to float for calculation
		#print("loss")
		#print(loss)
		
		#loss_v2= loss_fct(logits.view(-1,self.num_labels), labels.view(-1))
		
		#print("loss_v2")
		#print(loss_v2)
		
		return (loss, outputs) if return_outputs else loss	
	
