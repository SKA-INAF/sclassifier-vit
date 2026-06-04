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

# - SKLEARN
from sklearn import metrics
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, recall_score, precision_score
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import hamming_loss

# - TORCH
import torch
from torch.optim import AdamW
import torchvision.transforms.functional as TF
import torchvision.transforms as T

# - TRANSFORMERS
import transformers
from transformers import pipeline
from transformers import AutoProcessor, AutoModel
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import ViTForImageClassification, ViTConfig
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
from sclassifier_vit.dataset import get_multi_label_target_maps, get_single_label_target_maps
from sclassifier_vit.dataset import MultiLabelDataset, SingleLabelDataset
from sclassifier_vit.trainer import CustomTrainer
from sclassifier_vit.custom_transforms import (
	FlippingTransform, 
	Rotate90Transform, 
	RandomCenterCrop,
	MinMaxNormalization
)
from sclassifier_vit.metrics import build_multi_label_metrics, build_single_label_metrics
from sclassifier_vit import logger

# - Configure transformer logging
transformers.utils.logging.set_verbosity(transformers.logging.DEBUG)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()
    
# - Configure wandb
os.environ["WANDB_PROJECT"]= "sclassifier-vit"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints


#### GET SCRIPT ARGS ####
def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

###########################
##     ARGS
###########################
def get_args():
	"""This function parses and return arguments passed in"""
	parser = argparse.ArgumentParser(description="Parse args.")

	# - Input options
	parser.add_argument('-inputfile','--inputfile', dest='inputfile', required=False, type=str, default="", help='Input image (FITS/PNG). Takes precedence over --datalist') 
	parser.add_argument('-datalist','--datalist', dest='datalist', required=False, type=str, default="", help='Input data json filelist') 
	parser.add_argument('-datalist_cv','--datalist_cv', dest='datalist_cv', required=False, default="", type=str, help='Input data json filelist for validation') 

	# - Image pre-processing options
	parser.add_argument('--zscale', dest='zscale', action='store_true',help='Apply zscale transform to input images (default=false)')	
	parser.set_defaults(zscale=False)
	parser.add_argument('-zscale_contrast', '--zscale_contrast', dest='zscale_contrast', required=False, type=float, default=0.25, action='store', help='zscale contrast parameter (default=0.25)')
	parser.add_argument('--grayscale', dest='grayscale', action='store_true',help='Load input images in grayscale (1 chan tensor) (default=false)')	
	parser.set_defaults(grayscale=False)
	parser.add_argument('--resize', dest='resize', action='store_true', help='Resize input image before model processor. If false the model processor will resize anyway to its image size (default=false)')	
	parser.set_defaults(resize=False)
	parser.add_argument('-resize_size', '--resize_size', dest='resize_size', required=False, type=int, default=224, action='store', help='Resize size in pixels used if --resize option is enabled (default=224)')
	parser.add_argument('-norm_min', '--norm_min', dest='norm_min', required=False, type=float, default=0.0, action='store',help='Min normalization value in MinMaxNormalization transform (default=0.0)')
	parser.add_argument('-norm_max', '--norm_max', dest='norm_max', required=False, type=float, default=1.0, action='store',help='Max normalization value in MinMaxNormalization transform (default=1.0)')
	parser.add_argument('--cast_to_float', dest='cast_to_float', action='store_true', help='Cast input pixel_values to float in collate (default=cast)')
	parser.add_argument('--no_cast_to_float', dest='cast_to_float', action='store_false', help='Do not cast input pixel_values to float in collate (default=cast)')
	parser.set_defaults(cast_to_float=True)
	
	parser.add_argument('--reset_meanstd', dest='reset_meanstd', action='store_true', help='Reset mean/std parameters used in image processor (default=reset to mean=0, std=1)')
	parser.add_argument('--no_reset_meanstd', dest='reset_meanstd', action='store_false', help='Leave mean/std parameters set as provided in image processor (default=reset to mean=0, std=1)')
	parser.set_defaults(reset_meanstd=True)
	
	parser.add_argument('--use_model_processor', dest='use_model_processor', action='store_true', help='Use model image processor in data collator (default=false)')	
	parser.set_defaults(use_model_processor=False)
	
	# - Run options
	parser.add_argument('--predict', dest='predict', action='store_true', help='Predict model on input data (default=false)')	
	parser.set_defaults(predict=False)
	parser.add_argument('--test', dest='test', action='store_true', help='Run model test on input data (default=false)')	
	parser.set_defaults(test=False)
	parser.add_argument('--plot', dest='plot', action='store_true', help='Plot input data. Useful for debugging data transform/augmentation (default=false)')	
	parser.set_defaults(plot=False)
	parser.add_argument('-ngpu', '--ngpu', dest='ngpu', required=False, type=int, default=1, action='store',help='Number of gpus used for the run. Needed to compute the global number of training steps (default=1)')
	parser.add_argument('-binary_thr', '--binary_thr', dest='binary_thr', required=False, type=float, default=0.5, action='store',help='Binary selection threshold (default=0.5).')
	
	# - Model options
	parser.add_argument('--vitloader', dest='vitloader', action='store_true', help='If enabled use ViTForImageClassification to load model otherwise AutoModelForImageClassification (default=false)')	
	parser.set_defaults(vitloader=False)
	
	parser.add_argument('-model', '--model', dest='model', required=False, type=str, default="google/siglip-so400m-patch14-384", action='store', help='Model pretrained file name or weight path to be loaded {google/siglip-large-patch16-256, google/siglip-base-patch16-256, google/siglip-base-patch16-256-i18n, google/siglip-so400m-patch14-384, google/siglip-base-patch16-224}')
	
	parser.add_argument('-nepochs', '--nepochs', dest='nepochs', required=False, type=int, default=1, action='store',help='Number of epochs used in network training (default=100)')	
	#parser.add_argument('-optimizer', '--optimizer', dest='optimizer', required=False, type=str, default='adamw', action='store',help='Optimizer used (default=rmsprop)')
	parser.add_argument('-lr_scheduler', '--lr_scheduler', dest='lr_scheduler', required=False, type=str, default='constant', action='store',help='Learning rate scheduler used {constant, linear, cosine, cosine_with_min_lr} (default=constant)')
	parser.add_argument('-lr', '--lr', dest='lr', required=False, type=float, default=5e-5, action='store',help='Learning rate (default=5e-5)')
	parser.add_argument('-min_lr', '--min_lr', dest='min_lr', required=False, type=float, default=1e-6, action='store',help='Learning rate min used in cosine_with_min_lr (default=1.e-6)')
	parser.add_argument('-warmup_ratio', '--warmup_ratio', dest='warmup_ratio', required=False, type=float, default=0.2, action='store',help='Warmup ratio par (default=0.2)')
	parser.add_argument('-batch_size', '--batch_size', dest='batch_size', required=False, type=int, default=8, action='store',help='Batch size used in training (default=8)')
	parser.add_argument('-batch_size_eval', '--batch_size_eval', dest='batch_size_eval', required=False, type=int, default=None, action='store',help='Batch size used for evaluation. If None set equal to train batch size (default=None)')
	
	parser.add_argument('--drop_last', dest='drop_last', action='store_true',help='Drop last incomplete batch (default=false)')	
	parser.set_defaults(drop_last=False)
	parser.add_argument('-weight_decay','--weight_decay', dest='weight_decay', type=float, default=0.0, help='AdamW weight decay (default=0.0)')
	parser.add_argument('--head_dropout', dest='head_dropout', type=float, default=0.0, help='Dropout prob before classifier heads (default=0.0)')
	
	parser.add_argument('--ddp_find_unused_parameters', dest='ddp_find_unused_parameters', action='store_true', help='Flag passed to DistributedDataParallel when using distributed training (default=false)')	
	parser.set_defaults(ddp_find_unused_parameters=False)
	parser.add_argument('--fp16', dest='fp16', action='store_true', help='Enable fp16 (default=false)')	
	parser.set_defaults(fp16=False)
	parser.add_argument('--bf16', dest='bf16', action='store_true', help='Enable bf16 (default=false)')	
	parser.set_defaults(bf16=False)
	
	parser.add_argument("--num_workers", type=int, default=0)
	parser.add_argument("--pin_memory", type=str, choices=["true","false"], default="false")
	parser.add_argument("--persistent_workers", type=str, choices=["true","false"], default="false")
	
	# - Model loss options
	parser.add_argument("--use_weighted_loss", dest='use_weighted_loss', action="store_true", default=False, help="Use class-weighted loss (CE or focal alpha).")
	parser.add_argument("--weight_compute_mode", dest='weight_compute_mode', type=str, choices=["balanced", "inverse", "inverse_v2"], default="balanced", help="How to compute class weights")
	parser.add_argument('--normalize_weights', dest='normalize_weights', action='store_true', help="Enable normalization of class weights.")
	parser.add_argument('--no_normalize_weights', dest='normalize_weights', action='store_false', help="Disable normalization of class weights.")
	parser.set_defaults(normalize_weights=True)
	parser.add_argument("--loss_type", dest='loss_type', type=str, choices=["ce", "focal", "sol"], default="ce", help="Loss type: standard cross-entropy, focal loss, or score-oriented loss")
	parser.add_argument("--focal_gamma", dest='focal_gamma', type=float, default=2.0, help="Focal loss gamma (focusing parameter).")
	parser.add_argument("--set_focal_alpha_to_mild_estimate", dest='set_focal_alpha_to_mild_estimate', action="store_true", default=False, help="Set focal alpha to mild estimate, otherwise to class_weights.")
	parser.add_argument('-sol_score', '--sol_score', dest='sol_score', choices=["accuracy", "precision", "recall", "specificity", "f1", "tss", "csi", "hss1", "hss2"], required=False, type=str, default='tss', action='store', help='Solar score used (default=tss)')
	parser.add_argument('-sol_distribution', '--sol_distribution', dest='sol_distribution', choices=["uniform", "cosine"], required=False, type=str, default='uniform', action='store', help='Solar score distribution used (default=uniform)')
	parser.add_argument('-sol_mode', '--sol_mode', dest='sol_mode', choices=["weighted", "average"], required=False, type=str, default='average', action='store', help='Solar score averaging mode used (default=average)')
	parser.add_argument("--sol_add_constant", dest='sol_add_constant', action="store_true", default=False, help="Add constant (+1) to solar loss (default=false).")
		
	# - Image augmentations options
	parser.add_argument('--augmentation', dest='augmentation', action='store_true', help='Apply augmentation to train/val/test data (default=apply augmentation)')
	parser.add_argument('--no_augmentation', dest='augmentation', action='store_false', help='Do not apply augmentation to train/val/test data (default=apply augmentation)')
	parser.set_defaults(augmentation=True)
	
	parser.add_argument('--add_center_crop_augm', dest='add_center_crop_augm', action='store_true', help='If enabled, add center crop  augmentation in training (default=false)')	
	parser.set_defaults(add_rand_center_crop_augm=False)
	parser.add_argument('-crop_size', '--crop_size', dest='crop_size', required=False, type=int, default=224, action='store', help='Crop size (in pixels) used for center crop augmentation (default=224).')
	parser.add_argument('--add_rand_center_crop_augm', dest='add_rand_center_crop_augm', action='store_true', help='If enabled, add random center crop and resize (--resize_size) augmentation in training (default=false)')	
	parser.set_defaults(add_rand_center_crop_augm=False)
	parser.add_argument('-min_crop_fract', '--min_crop_fract', dest='min_crop_fract', required=False, type=float, default=0.65, action='store', help='Mininum crop fraction in random center crop (default=0.65).')
	parser.add_argument('--add_blur_augm', dest='add_blur_augm', action='store_true', help='If enabled, add gaussian blur augmentation with prob= (default=false)')	
	parser.set_defaults(add_blur_augm=False)
	parser.add_argument('-blur_prob', '--blur_prob', dest='blur_prob', required=False, type=float, default=0.1, action='store', help='Blur probability (default=0.1).')
	
	# - Evaluation/logging
	parser.add_argument('--skip_first_class', dest='skip_first_class', action='store_true',help='Skip first class (e.g. NONE/BACKGROUND) in multilabel classifier (default=false)')	
	parser.set_defaults(skip_first_class=False)
	
	parser.add_argument('--run_eval_on_start', dest='run_eval_on_start', action='store_true',help='Run model evaluation on start for debug (default=false)')	
	parser.set_defaults(run_eval_on_start=False)
	parser.add_argument('-logging_steps', '--logging_steps', dest='logging_steps', required=False, type=int, default=1, action='store',help='NUmber of logging steps (default=1)')
	parser.add_argument('--run_eval_on_step', dest='run_eval_on_step', action='store_true',help='Run model evaluation after each step (default=false)')	
	parser.set_defaults(run_eval_on_step=False)
	parser.add_argument('-gradient_accumulation_steps', '--gradient_accumulation_steps', dest='gradient_accumulation_steps', required=False, type=int, default=1, action='store',help='Number of updates steps to accumulate the gradients for, before performing a backward/update pass (default=1)')
	
	parser.add_argument('--freeze_backbone', dest='freeze_backbone', action='store_true',help='Make backbone layers are non-tranable (default=false)')	
	parser.set_defaults(freeze_backbone=False)
	parser.add_argument('-max_freeze_layer_id', '--max_freeze_layer_id', dest='max_freeze_layer_id', required=False, type=int, default=-1, action='store',help='ID of the last layer kept frozen. -1 means all are frozen if --freeze_backbone option is enabled (default=-1)')
	
	parser.add_argument('--multilabel', dest='multilabel', action='store_true', help='Do multilabel classification (default=false)')	
	parser.set_defaults(multilabel=False)
	parser.add_argument('-label_schema', '--label_schema', dest='label_schema', required=False, type=str, default='morph_tags', action='store',help='Predefined label schema to be used. For multilabel class: {"morph_tags","morph_tags_B1"}. For single-label classification: {"morph_tags","morph_class","binary_qa","anomaly_class","rg_morph_binary","rg_morph"} (default=morph_tags)')
	parser.add_argument('-background_label', '--background_label', dest='background_label', required=False, type=str, default='BACKGROUND', action='store',help='Name of background class used in predict when skip_first_class is enabled (default=BACKGROUND)')
	parser.add_argument('--binary', dest='binary', action='store_true',help='Choose binary classification label scheme (default=false)')	
	parser.set_defaults(binary=False)
	parser.add_argument('-metric_for_best_model', '--metric_for_best_model', dest='metric_for_best_model', required=False, type=str, default='loss', action='store', help='Metric used to select the best model {"loss","f1score", "f1score_micro", "recall", "precision"} (default=loss)')
	
	# - Run options
	parser.add_argument('-device', '--device', dest='device', required=False, type=str, default="cuda:0", action='store',help='Device identifier')
	parser.add_argument('-runname', '--runname', dest='runname', required=False, type=str, default="llava_1.5_radio", action='store',help='Run name')
	parser.add_argument('--verbose', dest='verbose', action='store_true',help='Enable verbose printout (default=false)')	
	parser.set_defaults(verbose=False)
	parser.add_argument("--report_to", dest='report_to', type=str, default="wandb", help="Report logs/metrics to {wandb, none}")
	parser.add_argument('-seed', '--seed', dest='seed', required=False, type=int, default=42, action='store',help='Random seed that will be set at the beginning of training (default=42)')
	
	# - Output options
	parser.add_argument('-outdir','--outdir', dest='outdir', required=False, default="", type=str, help='Output data dir') 
	parser.add_argument('-max_checkpoints', '--max_checkpoints', dest='max_checkpoints', required=False, type=int, default=1, action='store',help='Max number of saved checkpoints (default=1)')
	parser.add_argument('-outfile','--outfile', dest='outfile', required=False, default="classifier_results.json", type=str, help='Output file with saved inference results') 
	
	parser.add_argument('--save_base_path', dest='save_base_path', action='store_true', help='Save input base filename in output json catalog rather than full path (default=save full path)')	
	parser.set_defaults(save_base_path=False)
	
	args = parser.parse_args()	

	return args		
	
		
##################################
##    MODEL LOADER
##################################
def load_image_model_vit(
	args,
	id2label,
	label2id,
	num_labels,
	nclasses,
	inference_mode=False
):
	""" Load image model & processor (ViT loader version) """

	if inference_mode:
		# - Load model for inference
		model= ViTForImageClassification.from_pretrained(args.model)
		
		# Ensure head matches the checkpoint if it was trained with Dropout+Linear
		try:
			state = load_state_dict_any(args.model)  # works with checkpoint dir or .bin/.safetensors
			needs_seq_head = any(k.startswith("classifier.1.") for k in state.keys())
			has_plain_linear = hasattr(model, "classifier") and isinstance(model.classifier, torch.nn.Linear)
			if needs_seq_head and has_plain_linear:
				p = getattr(getattr(model, "config", object()), "head_dropout", args.head_dropout)
				maybe_wrap_classifier_with_dropout(model, p)
				# reload so classifier.* weights map correctly
				model.load_state_dict(state, strict=False)
		except Exception as e:
			logger.warning(f"Classifier head alignment skipped (non-fatal): {e}")
		
		model.eval()
		
	else:
		# - Load model for training
		if args.binary:
			config= ViTConfig.from_pretrained(
				args.model,
				problem_type=None, 
				num_labels=1
			)	
		else:
			config= ViTConfig.from_pretrained(
				args.model,
				problem_type="multi_label_classification" if args.multilabel else "single_label_classification", 
				id2label=id2label, 
				label2id=label2id,
				num_labels=num_labels
			)
		
		model= ViTForImageClassification.from_pretrained(
			args.model,
			config=config
		)
		
		# - Replace the head with 1 logit for binary class
		if args.binary:
			in_features = model.classifier.in_features
			model.classifier = torch.nn.Linear(in_features, 1)
			model.config.num_labels = 1  # avoid confusion; we’ll provide our own loss
			model.config.problem_type = None  # don't let HF pick MSE; we handle loss in Trainer
					
		# - Add dropout layer in architecture and config?
		maybe_wrap_classifier_with_dropout(
			model, args.head_dropout, num_out=(1 if args.binary else num_labels)
		)
		if hasattr(model, "config"):
			setattr(model.config, "head_dropout", float(args.head_dropout))	
				
	# - Load processor	
	image_processor = ViTImageProcessor.from_pretrained(args.model)
	
	return model, image_processor



def load_image_model_auto(
	args,
	id2label,
	label2id,
	num_labels,
	nclasses,
	inference_mode=False
):
	""" Load image model & processor (AutoModelForImageClassification loader version) """

	if inference_mode:
		# - Load model for inference
		model = AutoModelForImageClassification.from_pretrained(args.model)
		
		# - Ensure head matches the checkpoint if it was trained with Dropout+Linear
		try:
			state = load_state_dict_any(args.model)  # works with checkpoint dir or .bin/.safetensors
			needs_seq_head = any(k.startswith("classifier.1.") for k in state.keys())
			has_plain_linear = hasattr(model, "classifier") and isinstance(model.classifier, torch.nn.Linear)
			if needs_seq_head and has_plain_linear:
				p = getattr(getattr(model, "config", object()), "head_dropout", args.head_dropout)
				maybe_wrap_classifier_with_dropout(model, p)
				# reload so classifier.* weights map correctly
				model.load_state_dict(state, strict=False)
		except Exception as e:
			logger.warning(f"Classifier head alignment skipped (non-fatal): {e}")
		
		model.eval()
		
	else:
		
		# - Load standard-head model
		if args.binary:		
			# - Load standard tmp model
			model = AutoModelForImageClassification.from_pretrained(args.model, num_labels=2)  # temp
				
			# - Replace the head with 1 logit
			in_features = model.classifier.in_features
			model.classifier = torch.nn.Linear(in_features, 1)
			model.config.num_labels = 1  # avoid confusion; we'll provide our own loss
			model.config.problem_type = None  # don't let HF pick MSE; we handle loss in Trainer	
					
		else:
			# - Load standard model
			try:
				model = AutoModelForImageClassification.from_pretrained(
					args.model, 
					problem_type="multi_label_classification" if args.multilabel else "single_label_classification", 
					id2label=id2label, 
					label2id=label2id,
					num_labels=num_labels
				)
			except Exception as e:
				logger.warning(f"Failed to load model {args.model} (err={str(e)}), retrying with ignore_mismatched_sizes=True ...")
				model = AutoModelForImageClassification.from_pretrained(
					args.model, 
					problem_type="multi_label_classification" if args.multilabel else "single_label_classification", 
					id2label=id2label, 
					label2id=label2id,
					num_labels=num_labels,
					ignore_mismatched_sizes=True
				)
			
			
		# - Add dropout in architecture & config?	
		maybe_wrap_classifier_with_dropout(
			model, args.head_dropout, num_out=(1 if args.binary else num_labels)
		)
		if hasattr(model, "config"):
			setattr(model.config, "head_dropout", float(args.head_dropout))
		
	# - Load processor	
	image_processor = AutoImageProcessor.from_pretrained(args.model)
		
	return model, image_processor	


def load_model(
	args,
	id2label,
	label2id,
	num_labels,
	nclasses,
	inference_mode=False
):
	""" Load image model & processor """

	#===================================
	#==     VIT-LOADER
	#===================================
	# - Load model
	if args.vitloader:
		model, image_processor= load_image_model_vit(
			args,	
			id2label,
			label2id,
			num_labels,
			nclasses,
			inference_mode
		)
	
	#===================================
	#==     AUTOMODEL
	#===================================
	else:
		model, image_processor= load_image_model_auto(
			args,	
			id2label,
			label2id,
			num_labels,
			nclasses,
			inference_mode
		)
		
	return model, image_processor	
				

def freeze_model(model, args):
	""" Freeze model backbone """
	
	logger.info("Freezing base layers ...")
		
	# 1. Check model type
	model_type = "resnet" if "ResNet" in type(model).__name__ else "vit"
		
	# 2. Build the ResNet map dynamically if needed
	resnet_registry = build_resnet_layer_registry(model) if model_type == "resnet" else None
		
	# 3. Freeze layers
	for name, param in model.base_model.named_parameters():
		# - Select encoder layers
		is_backbone_layer = (
			name.startswith("vision_model.encoder") or # VIT MODELS
			name.startswith("vision_model.embeddings") or # VIT MODELS (EMBEDDINGS)
			name.startswith("encoder.stages") or # RESNET MODELS
			name.startswith("embedder") # # RESNET MODELS (STEM)
		)
			
		if not is_backbone_layer:
			continue
			
		# - Freeze usign layer_id threshold
		#layer_index= extract_layer_id_vit(name)
		layer_index = extract_layer_id(name, model_type=model_type, resnet_registry=resnet_registry)
		logger.debug(f"Backbone layer {name}: index={layer_index} ...")
			
		if layer_index != -1:
			if args.max_freeze_layer_id == -1 or (args.max_freeze_layer_id >= 0 and layer_index < args.max_freeze_layer_id):
				logger.info(f"--> Freezing backbone layer {name} (index={layer_index}) ...")
				param.requires_grad = False
				
		# 4. Handle structural shortcuts for larger models (ResNet-50/101/152)	
		elif "shortcut" in name and model_type == "resnet":
			match = re.search(r"stages\.(\d+)\.layers\.(\d+)", name)
			logger.debug(f"Shortcut Backbone layer {name}: index={layer_index}, match={match} ...")
				
			if match:
				# Build the exact dictionary key path used by the companion convolution layer
				companion_layer_key = f"encoder.stages.{match.group(1)}.layers.{match.group(2)}.layer.0.convolution.weight"
        
				# Check if this exact companion name is registered in our registry dictionary
				if companion_layer_key in resnet_registry:
					companion_idx = resnet_registry[companion_layer_key]
            
					# Apply your freezing threshold criteria
					if args.max_freeze_layer_id == -1 or (args.max_freeze_layer_id >= 0 and companion_idx < args.max_freeze_layer_id):
						param.requires_grad = False
						logger.info(f"--> Freezing shortcut layer {name} (tracked to companion index={companion_idx}) ...")

	# - Print resulting model		
	logger.info("Print base model info ...")	
	for name, param in model.base_model.named_parameters():
		logger.info(f"Layer {name} requires_grad? {param.requires_grad}")
				
	logger.info("Print entire model info ...")
	for name, param in model.named_parameters():
		logger.info(f"Layer {name} requires_grad? {param.requires_grad}")
		
	return model


##################################
##     TRAINING OPTIONS
##################################
def load_training_opts(args):
	""" Prepare training options """
			
	# - Set output dir
	output_dir= args.outdir
	if output_dir=="":
		output_dir= os.getcwd()
			
	log_dir= os.path.join(output_dir, "logs/")
	
	# - Set eval & save strategy
	eval_strategy= "no"
	load_best_model_at_end= False
	save_strategy= "no"
	batch_size_eval= args.batch_size if args.batch_size_eval is None else args.batch_size_eval
	if args.datalist_cv!="":
		load_best_model_at_end= True
		if args.run_eval_on_step:
			eval_strategy= "steps"
			save_strategy= "steps"
		else:
			eval_strategy= "epoch"
			save_strategy= "epoch"
			
	# - Set training options
	logger.info("Set model options ...")
	training_opts= transformers.TrainingArguments(
		output_dir=output_dir,
		do_train=True if not args.test else False,
		do_eval=True if not args.test and args.datalist_cv!="" else False,
		do_predict=True if args.test else False,
		num_train_epochs=args.nepochs,
		optim="adamw_torch",
		lr_scheduler_type=args.lr_scheduler,
		learning_rate=args.lr,
		warmup_ratio=args.warmup_ratio,
		#warmup_steps=num_warmup_steps,
		per_device_train_batch_size=args.batch_size,
		per_device_eval_batch_size=batch_size_eval,
		gradient_accumulation_steps=args.gradient_accumulation_steps,
		dataloader_drop_last= args.drop_last,
		eval_strategy=eval_strategy,
		eval_on_start=args.run_eval_on_start,
		eval_steps=args.logging_steps,
		metric_for_best_model=args.metric_for_best_model,
		greater_is_better=True,
		load_best_model_at_end=load_best_model_at_end,
		##batch_eval_metrics=False,
		##label_names=label_names,# DO NOT USE (see https://discuss.huggingface.co/t/why-do-i-get-no-validation-loss-and-why-are-metrics-not-calculated/32373)
		#save_strategy="epoch" if args.save_model_every_epoch else "no",
		save_strategy=save_strategy,
		save_total_limit=args.max_checkpoints, # at most keep only BEST + LAST
		logging_dir = log_dir,
		log_level="debug",
		logging_strategy="steps",
		logging_first_step=True,
		logging_steps=args.logging_steps,
		logging_nan_inf_filter=False,
		#disable_tqdm=True,
		run_name=args.runname,
		#report_to="wandb",  # enable logging to W&B
		report_to=args.report_to,
		seed=args.seed,
		dataloader_num_workers=args.num_workers,
		dataloader_pin_memory=(args.pin_memory=="true"),
		dataloader_persistent_workers=(args.persistent_workers=="true" and args.num_workers>0),
		ddp_find_unused_parameters=args.ddp_find_unused_parameters,
		fp16=args.fp16,
		bf16=args.bf16,
		weight_decay=args.weight_decay,
		#remove_unused_columns=False if args.data_modality=="multimodal" else True,
	)
	
	print("--> training options")
	print(training_opts)		
			
	return training_opts		

def load_optimizer(model, dataset, args):
	""" Build and return optimizer/lr scheduler """

	# - Create optimizer
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
	
	# - Compute params
	nsamples= dataset.get_sample_size()
	tot_batch_size= args.ngpu * args.batch_size * args.gradient_accumulation_steps
	if args.drop_last:
		n_batches = nsamples // args.batch_size
	else:
		n_batches= math.ceil(nsamples / args.batch_size)
                    
	num_update_steps_per_epoch = max(n_batches // args.gradient_accumulation_steps + int(n_batches % args.gradient_accumulation_steps > 0), 1)
	max_steps = math.ceil(args.nepochs * num_update_steps_per_epoch)
	training_steps= max_steps
	warmup_steps = math.ceil(training_steps * args.warmup_ratio)
	
	logger.info(f"Train pars: nsamples={nsamples}, epochs={args.nepochs}, batch_size={args.batch_size}, gradacc={args.gradient_accumulation_steps}, tot_batch_size={tot_batch_size}, n_batches={n_batches}, num_update_steps_per_epoch={num_update_steps_per_epoch}, max_steps={max_steps}, steps={training_steps}, warmup_steps={warmup_steps}")
	
	# - Create lr scheduler
	if args.lr_scheduler=="constant":
		logger.info("Setting constant scheduler ...")
		scheduler= transformers.get_constant_schedule(optimizer)
	elif args.lr_scheduler=="linear":
		logger.info("Setting linear scheduler with warmup ...")
		scheduler= transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps)
	elif args.lr_scheduler=="cosine":
		logger.info("Setting cosine scheduler with warmup ...")
		scheduler= transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps)
	elif args.lr_scheduler=="cosine_with_min_lr":
		logger.info("Setting cosine scheduler with warmup & min lr ...")
		scheduler= transformers.get_cosine_with_min_lr_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps, min_lr=args.min_lr)
	else:
		logger.info("Setting constant scheduler ...")
		scheduler= transformers.get_constant_schedule(optimizer)	

	return optimizer, scheduler

##################################
##     DATA TRANSFORMS
##################################
def load_image_transform(args, image_processor):
	""" Load image data transform """

	# - Create data transforms
	logger.info("Creating data transforms ...")
	
	if "height" in image_processor.size and "width" in image_processor.size:
		size = (image_processor.size["height"], image_processor.size["width"])
		
	elif "shortest_edge" in image_processor.size:
		# Fallback for models using shortest_edge (squares like 224x224)
		dim = image_processor.size["shortest_edge"]
		size = (dim, dim)
	else:
		raise RuntimeError("Cannot find height/width or shortest_edge in processor, please check it!")
	
	mean_proc = image_processor.image_mean
	std_proc = image_processor.image_std
	do_resize= getattr(image_processor, "do_resize", None)
	do_rescale= getattr(image_processor, "do_rescale", None)
	rescale_factor= getattr(image_processor, "rescale_factor", None)
	do_normalize= getattr(image_processor, "do_normalize", None)
	do_convert_rgb= getattr(image_processor, "do_convert_rgb", None)
	mean_reset= [0.,0.,0.]
	std_reset= [1.,1.,1.]
	if args.reset_meanstd:
		mean= mean_reset
		std= std_reset
	else:
		mean= mean_proc
		std= std_proc
		
	print("*** Image processor config pars ***")
	print(f"do_resize? {do_resize}")
	print("size: ", (size))
	print(f"do_rescale? {do_rescale}")
	print(f"rescale_factor: {rescale_factor}")
	print(f"do_normalize? {do_normalize}")
	print("mean_proc: ", (mean_proc))
	print("std_proc: ", (std_proc))
	print("mean: ", (mean))
	print("std: ", (std))
	print(f"do_convert_rgb? {do_convert_rgb}")
		
	sigma_min= 1.0
	sigma_max= 3.0
	ksize= 3.3 * sigma_max
	kernel_size= int(max(ksize, 5)) # in imgaug kernel_size viene calcolato automaticamente dalla sigma così, ma forse si può semplificare a 3x3
	blur_transf= T.GaussianBlur(kernel_size, sigma=(sigma_min, sigma_max))
	blur_aug= T.RandomApply([blur_transf], p=args.blur_prob)
	center_crop_aug= T.CenterCrop(size=args.crop_size)
	rand_center_crop_aug= RandomCenterCrop(min_frac=args.min_crop_fract, max_frac=1.0, output_size=None)
	
	# - Set train augmentation
	transf_list= []
	
	if args.add_center_crop_augm: # add CenterCrop
		logger.info(f"Adding center crop augmentation (size={args.crop_size}) in train dataset ...")
		transf_list.append(center_crop_aug)
		
	if args.add_rand_center_crop_augm: # add RandomCenterCrop
		logger.info(f"Adding random center crop augmentation (min_frac={args.min_crop_fract}) in train dataset ...")
		transf_list.append(rand_center_crop_aug)
		
	transf_list.extend(
		[
			T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
			FlippingTransform(),
			Rotate90Transform(),
			MinMaxNormalization(norm_min=args.norm_min, norm_max=args.norm_max),
		]
	)
	
	if args.add_blur_augm:
		logger.info(f"Adding random blur augmentation (prob={args.blur_prob}) in train dataset ...")
		transf_list.append(blur_aug)

	transf_list.extend(
		[
			#T.ToTensor(),
			T.Normalize(mean=mean, std=std),
		]
	)

	transform_train = T.Compose(transf_list)
	
	# - Set test/validation transform
	transform_valtest = T.Compose(
		[
			T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
			MinMaxNormalization(norm_min=args.norm_min, norm_max=args.norm_max),
			#T.ToTensor(),
			T.Normalize(mean=mean, std=std),
		]
	)
	
	return transform_train, transform_valtest


######################
##   LOAD DATASET   ##
######################
def load_dataset(
	args, 
	image_processor,
	nclasses,
	id2target
):
	""" Load dataset """			
			
	#====================================
	#==   CREATE DATA TRANSFORMS
	#====================================
	# - Disable augmentation?
	transform_train= None
	transform_valtest= None
	if args.augmentation:
		transform_train, transform_valtest= load_image_transform(args, image_processor)
	else:
		logger.info("Setting all augmenter transforms (train/cv/test) to None (e.g. disabling augmentation) ...")
	
	#====================================
	#==   CREATE DATASET
	#====================================
	# - Init stuff
	dataset_cv= None
	dataset= None
	nsamples= 0
	nsamples_cv= 0
	DatasetClass= None
	
	if args.multilabel:
		DatasetClass= MultiLabelDataset
		class_type_str= "multi-label classification"
	else:
		DatasetClass= SingleLabelDataset
		class_type_str= "single-label classification"
	
	# - Create dataset
	# - TEST SET
	if args.predict or args.test:
		logger.info(f"Create dataset for prediction/test ({class_type_str}) ...")	
		dataset= DatasetClass(
			filename=args.datalist,
			transform=transform_valtest,
			load_as_gray=args.grayscale,
			apply_zscale=args.zscale, zscale_contrast=args.zscale_contrast,
			resize=args.resize, resize_size=args.resize_size,
			nclasses=nclasses,
			id2target=id2target,
			verbose=args.verbose,
		)
		nsamples= dataset.get_sample_size()
		logger.info(f"#{nsamples} entries in dataset ...")
	
	else:
		# - TRAIN SET
		logger.info(f"Create train dataset ({class_type_str}) ...")
		dataset= DatasetClass(
			filename=args.datalist,
			transform=transform_train,
			load_as_gray=args.grayscale,
			apply_zscale=args.zscale, zscale_contrast=args.zscale_contrast,
			resize=args.resize, resize_size=args.resize_size,
			nclasses=nclasses,
			id2target=id2target,
			verbose=args.verbose,
		)
		nsamples= dataset.get_sample_size()
		logger.info(f"#{nsamples} entries in train dataset ...")
		
		# - VALIDATION SET
		if args.datalist_cv!="":
			logger.info("Create val dataset (multi-label classification) ...")
			dataset_cv= DatasetClass(
				filename=args.datalist_cv,
				transform=transform_valtest,
				load_as_gray=args.grayscale,
				apply_zscale=args.zscale, zscale_contrast=args.zscale_contrast,
				resize=args.resize, resize_size=args.resize_size,
				nclasses=nclasses,
				id2target=id2target,
				verbose=args.verbose,
			)
			nsamples_cv= dataset_cv.get_sample_size()
			logger.info(f"#{nsamples_cv} entries in val dataset ...")
	
	return dataset, dataset_cv
			
##########################################
##    DATA COLLATORS
##########################################
def collate_fn(batch):
	pixel_values= []
	labels= []
	for item in batch:
		if item[0] is None:
			continue
		pixel_values.append(item[0])
		labels.append(item[1])
			
	if args.cast_to_float:
		pixel_values= torch.stack(pixel_values).float()
	else: 
		pixel_values= torch.stack(pixel_values)
	labels= torch.stack(labels)
	return {"pixel_values": pixel_values, "labels": labels}
		

class ImgDataCollator:
	def __init__(
		self, 
		image_processor=None, 
		do_resize=True, 
		do_normalize=True, 
		do_rescale=True,
		cast_to_float=True,
	):
		""" Define image data collator """
		self.processor = image_processor
		self.do_resize = do_resize
		self.do_normalize = do_normalize
		self.do_rescale= do_rescale
		self.cast_to_float= cast_to_float
	
	def __call__(self, batch):
		
		# - Collect batch items
		images, labels = [], []
		
		for item in batch:
			if isinstance(item, dict):
				img = item.get("pixel_values", item.get("image"))
				lab = item.get("labels", item.get("label"))
			else:  # tuple/list
				if not item or item[0] is None:
					continue
				img, lab = item[0], item[1]

			if img is None:
				continue
			
			images.append(img)
			labels.append(lab)
	
		if len(images) == 0:
			# Edge case: all items were None — return empty batch tensors
			return {
				"pixel_values": torch.empty(0),
				"labels": torch.empty(0, dtype=torch.long)
			}
	
		# - Set pixel values ---
		if self.processor is not None:
			# - Pass the raw list to the processor (PIL / np / torch supported)
			proc_out = self.processor(
				images,
				return_tensors="pt",
				do_resize=self.do_resize,       # set False if already resized
				do_normalize=self.do_normalize, # set False if already normalized
				do_rescale=self.do_rescale,     # set False if already rescaled
			)
			pixel_values = proc_out["pixel_values"]
			
		else:
			# Assume items are already tensors; ensure [B, C, H, W]
			images = [img if isinstance(img, torch.Tensor) else torch.as_tensor(img) for img in images] 
			pixel_values = torch.stack(images, dim=0)
	
		# - Set labels
		labels= torch.stack(labels)
		
		# - Cast to float?
		if self.cast_to_float:
			pixel_values= pixel_values.float()
	
		return {"pixel_values": pixel_values, "labels": labels}				
			
			
#############################
###      RUN TEST
#############################
def run_test(
	trainer,
	dataset,
	args
):
	""" Run test """
			
	predictions, labels, metrics= trainer.predict(dataset, metric_key_prefix="predict")
		
	print("--> predictions")
	print(type(predictions))
	print(predictions)
		
	print("--> labels")
	print(type(labels))
	print(labels)
		
	print("--> prediction metrics")
	print(metrics) 
		
	trainer.log_metrics("predict", metrics)
	trainer.save_metrics("predict", metrics)		
			
			
##############################
##     RUN PREDICT
##############################
def run_predict(
	model,
	dataset,
	args,
	id2label,
	image_processor=None,
	data_collator=None,
	device="cuda:0"
):
	""" Run model predict """
		
	inference_results= {"data": []}
	nsamples= dataset.get_sample_size()
	
	for i in range(nsamples):
		if i%1000==0:
			logger.info("#%d/%d images processed ..." % (i+1, nsamples))
		
		# - Retrieve image info
		image_info= dataset.load_image_info(i)
		sname= image_info["sname"]
							
		# - Load image & extract embeddings
		image_tensor= dataset.load_tensor(i)
		if image_tensor is None:
			logger.warning("Skip None tensor at index %d ..." % (i))
			continue
				
		image_tensor= image_tensor.unsqueeze(0).to(device)
				
		if args.cast_to_float:
			image_tensor= image_tensor.float()
				
		with torch.no_grad():
			outputs = model(image_tensor)
			logits = outputs.logits
							
  	# - Compute predicted labels & probs
		if args.multilabel:
			sigmoid = torch.nn.Sigmoid()
			sigmoid_thr = getattr(args, "binary_thr", 0.5)
			probs = sigmoid(logits.squeeze().cpu()).numpy()
			#probs_v2 = sigmoid(logits_v2.squeeze().cpu()).numpy()
			predictions = np.zeros(probs.shape)
			predictions[np.where(probs >= sigmoid_thr)] = 1 # turn predicted id's into actual label names
				
			predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
			predicted_probs = [float(probs[idx]) for idx, label in enumerate(predictions) if label == 1.0]
			if not predicted_labels and args.skip_first_class:
				#print("max(probs)")
				#print(max(probs))
				min_prob= 1 - max(probs)
				predicted_probs= [min_prob]
				predicted_labels= [args.background_label]	
					
			# - Fill prediction results in summary dict
			image_info["label_pred"]= list(predicted_labels)
			image_info["prob_pred"]= list([float(item) for item in predicted_probs])
				
			if args.verbose:
				print("== Image: %s ==" % (sname))
				print("--> logits")
				print(logits)
				print("--> probs")
				print(probs)
				#print("--> probs_v2")
				#print(probs_v2)
					
				print("--> predicted class id enc")
				print(predictions)
				print("--> predicted labels")
				print(predicted_labels)
				print("--> predicted probs")
				print(predicted_probs)
					
		else:
			softmax = torch.nn.Softmax(dim=0)
			probs = softmax(logits.squeeze().cpu()).numpy()
			class_id= np.argmax(probs)
			predicted_label = id2label[class_id]
			predicted_prob= probs[class_id]
				
			# - Fill prediction results in summary dict
			image_info["label_pred"]= str(predicted_label)
			image_info["prob_pred"]= float(predicted_prob)
					
			if args.verbose:
				print("== Image: %s ==" % (sname))
				print("logits.squeeze().cpu()")
				print(logits.squeeze().cpu())
				print(logits.squeeze().cpu().shape)
				print("--> probs")
				print(probs)
				print("--> predicted class id")
				print(class_id)
				print("--> predicted label")
				print(predicted_label)
				print("--> predicted probs")
				print(predicted_prob)
			
		# - Modify filepath in output file
		if "filepaths" in image_info:
			if args.inputfile!="":
				fname= image_info["filepaths"][0]
				fname_base= os.path.basename(os.path.abspath(fname))
				if args.save_base_path:	
					del image_info["filepaths"]
					image_info["filepath"]= fname_base
				else:
					del image_info["filepaths"]
					image_info["filepath"]= fname
			else:
				if args.save_base_path:	
					filepaths_mod= []
					for fname in image_info["filepaths"]:
						fname_base= os.path.basename(os.path.abspath(fname))
						filepaths_mod.append(fname_base)
					image_info["filepaths"]= filepaths_mod
			
		# - Append inference data to list
		inference_results["data"].append(image_info)
		
	# - Remove "data" key for single-image input
	if args.inputfile!="":
		inference_out_data= inference_results["data"][0]
	else:
		inference_out_data= inference_results
		
	# - Save json file
	logger.info("Saving inference results with prediction info to file %s ..." % (args.outfile))
	with open(args.outfile, 'w') as fp:
		json.dump(inference_out_data, fp, indent=2)
	
####################################
###     RUN PLOT
####################################
def run_plot(
	dataset,
	args,
	device="cuda:0"
):
	""" Make plots """
		
	nsamples= dataset.get_sample_size()
	
	for i in range(nsamples):
		if i%1000==0:
			logger.info("#%d/%d images processed ..." % (i+1, nsamples))

		# - Retrieve image info
		image_info= dataset.load_image_info(i)
		sname= image_info["sname"]
			
		# - Retrieve image tensor
		image_tensor= dataset.load_tensor(i)
		if image_tensor is None:
			logger.warning("Skip None tensor at index %d ..." % (i))
			continue
				
		image_tensor= image_tensor.to(device)
				
		if args.cast_to_float:
			image_tensor= image_tensor.float()

		print(f"--> image_tensor no. {i+1} ({sname}) ...")
		print("image_tensor.shape")
		print(image_tensor.shape)
			
		# - Convert to numpy
		print(f"--> image_npy no. {i+1} ({sname}) ...")
		image_npy= image_tensor.cpu().detach().permute(1, 2, 0).numpy()
		print("image_npy.shape")
		print(image_npy.shape)
		print("image_npy min/max")
		print(image_npy.min())
		print(image_npy.max())
			
		# - Convert to uint8?
		image_npy_min= image_npy.min()
		image_npy_max= image_npy.max()
		if image_npy_min==0 and image_npy_max==255:
			image_npy= image_npy.astype(np.uint8)
			
		# - Plot image
		print(f"--> Plotting image_npy no. {i+1} ({sname}) ...")
		plt.imshow(image_npy, cmap="inferno")
		plt.show()	
		
#####################################
##     RUN TRAIN
#####################################
def run_train(
	trainer,
	args
):
	""" Run model train """		
	
	#train_result = trainer.train(resume_from_checkpoint=checkpoint)
	train_result = trainer.train()
	
	# - Save model
	logger.info("Saving trained model ...")	
	trainer.save_model()

	# - Save metrics
	logger.info("Saving train metrics ...")        
	trainer.log_metrics("train", train_result.metrics)
	trainer.save_metrics("train", train_result.metrics)
	trainer.save_state()
	
	print("train metrics")
	print(train_result.metrics) 
        
	# - Run evaluation
	if args.datalist_cv!="":
		logger.info("Running model evaluation ...")
		metrics = trainer.evaluate()
		trainer.log_metrics("eval", metrics)
		trainer.save_metrics("eval", metrics)
		print("eval metrics")
		print(metrics) 		
		
##############
##   MAIN   ##
##############
def main():
	"""Main function"""

	#===========================
	#==   PARSE ARGS
	#===========================
	logger.info("Get script args ...")
	try:
		args= get_args()
	except Exception as ex:
		logger.error("Failed to get and parse options (err=%s)",str(ex))
		return 1

	# - Read args
	if args.inputfile=="" and args.datalist=="":
		logger.error("Empty inputfile and datalist args, you must provide at least one!")
		return 1
		
	# - Set options
	if args.inputfile!="":
		logger.info(f"Overriding datalist with inputfile {inputfile} ...")
		args.datalist= args.inputfile
	
	device = torch.device(args.device if torch.cuda.is_available() else "cpu")
		
	# - Set class options
	if args.multilabel:
		id2label, label2id, id2target= get_multi_label_target_maps(args.label_schema, args.skip_first_class)
	else:
		id2label, label2id, id2target= get_single_label_target_maps(args.label_schema)
		
	#nclasses= len(id2target)   # - This counts all classes (e.g. 6)
	num_labels= len(id2label)  # - If skip_first_class, this is =5
	nclasses= num_labels
	label_names= list(label2id.keys())
	
	print("id2label")
	print(id2label)
	print("label2id")
	print(label2id)
	print("id2target")
	print(id2target)
	print("num_labels")
	print(num_labels)
	print("nclasses")
	print(nclasses)

	##################################
	##     MODEL
	##################################
	# - Load model & processor
	inference_mode= False
	if args.test or args.predict:
		inference_mode= True
	model, image_processor= load_model(args, id2label, label2id, num_labels, nclasses, inference_mode)
	
	# - Move model to device
	model= model.to(device)
	
	print("*** MODEL ***")
	print(model)
	print("")
	
	# - Freeze backbone?
	if args.freeze_backbone:
		logger.info("Freezing model base layers ...")
		model= freeze_model(model, args)
		
	##################################
	##     DATASET
	##################################
	# - Create datasets
	logger.info("Creating datasets ...")
	dataset, dataset_cv= load_dataset(
		args, 
		image_processor,
		nclasses,
		id2target
	)
 
	# - Create data collator
	logger.info("Creating data collator ...")
	data_collator= ImgDataCollator(
		image_processor=image_processor if args.use_model_processor else None, 
		do_resize=image_processor.do_resize if args.use_model_processor else False,                   # set to True only if processor should resize
		do_normalize=image_processor.do_normalize if args.use_model_processor else False,              # set to True only if processor should normalize
		do_rescale=image_processor.do_rescale if args.use_model_processor else False                  # set to True only if processor should rescale
	)

	#######################################
	##     SET TRAINER
	#######################################
	# - Training options
	logger.info("Creating training options ...")
	training_opts= load_training_opts(args)

	# - Set optimizer & scheduler
	logger.info("Creating optimizer/lr scheduler ...")
	optimizer, scheduler= load_optimizer(model, dataset, args)
	
	# - Set metrics
	logger.info("Creating metrics ...")
	if args.multilabel:
		compute_metrics_custom= build_multi_label_metrics(label_names)
	else:
		compute_metrics_custom= build_single_label_metrics(label_names)
		
	# - Compute class weights
	class_weights= None
	class_weights_binary= None
	focal_alpha= None
	
	if args.use_weighted_loss:
		logger.info("Computing class weights from dataset ...")
		class_weights = dataset.compute_class_weights(
			num_classes=num_labels, 
			id2target=id2target,
			scheme=args.weight_compute_mode,
			normalize=args.normalize_weights,
			binary=False
		)
		print("--> CLASS WEIGHTS")
		print(class_weights)
		
		# - Compute binary class weights?
		if args.binary: 
			logger.info("Computing binary class weights from dataset ...")
			class_weights_binary= dataset.compute_class_weights(
				num_classes=num_labels, 
				id2target=id2target,
				scheme=args.weight_compute_mode,
				normalize=args.normalize_weights,
				binary=True,
				positive_label=1, 
				laplace=1.0
			)
			
			print("--> BINARY CLASS WEIGHTS")
			print(class_weights_binary) 
		
		# Set focal loss pars
		#   - For focal alpha in multiclass, you can re-use class_weights
		#   - Often alpha ~ class_weights (normalized); you can also pass a float
		if args.loss_type=="focal":
			if args.set_focal_alpha_to_mild_estimate:
				logger.info("Setting focal alpha to mild estimate ...")
				focal_alpha, counts= dataset.compute_mild_focal_alpha_from_dataset(
    			num_classes=num_labels,
    			id2target=id2target,
					exponent=0.5,
					cap_ratio=10.0,
					device=device
				)
			else:
				logger.info("Setting focal alpha to class_weights if not None ...")
				counts= None
				focal_alpha = class_weights if args.loss_type == "focal" else None	

			def summarize_alpha(alpha_t, counts=None):
				if alpha_t is None:
					print("Focal alpha: None")
				else:
					alpha = alpha_t.detach().cpu().numpy()
					print("Focal alpha  :", np.round(alpha, 4).tolist(), " (mean=", round(alpha.mean(), 4), ", median=", round(np.median(alpha), 4), ")")
			
				if counts is not None:
					print("Class counts :", counts.tolist())
		
			print("--> FOCAL GAMMA")
			print(args.focal_gamma)
			print("--> FOCAL ALPHA")
			print(focal_alpha)
			summarize_alpha(focal_alpha, counts)
		
		
	# - Initialize trainer
	logger.info("Initialize model trainer ...")
	trainer= CustomTrainer(
		# - Standard args
		model=model,
		args=training_opts,
		train_dataset=dataset,
		eval_dataset=dataset_cv,
		compute_metrics=compute_metrics_custom,
		processing_class=image_processor,
		data_collator=data_collator, # collate_fn
		optimizers=(optimizer, scheduler),
		# - Custom args
		multilabel=args.multilabel,
		class_weights=class_weights,
		loss_type=args.loss_type, # "ce" or "focal"
		focal_gamma=args.focal_gamma,
		focal_alpha=focal_alpha,  # tensor[C] or float or None
		sol_score=args.sol_score,
		sol_distribution=args.sol_distribution,
		sol_mode=args.sol_mode,
		sol_add_constant=args.sol_add_constant,
		binary_pos_weights=class_weights_binary,
		logitout_size=(1 if args.binary else num_labels),
	)
	
	#######################################
	##     RUN TEST
	#######################################		
	# - Run predict on test set
	if args.test:
		logger.info("Predict model on input data %s ..." % (args.datalist))
		run_test(trainer, dataset, args)
	
	################################
	##    RUN PREDICT
	################################
	# - Run predict
	elif args.predict:
		logger.info("Running model inference on input data %s ..." % (args.datalist))
		run_predict(model, dataset, args, id2label, image_processor, data_collator=data_collator, device=device)
	
	################################
	##    PLOT DATASET
	################################
	# - Draw images
	elif args.plot:
		logger.info("Plotting input data %s ..." % (args.datalist))
		run_plot(dataset, args, device=device)
		
	################################
	##    TRAIN
	################################
	# - Run model train
	else:
		logger.info("Run model training ...")
		run_train(trainer, args)	
		
	return 0
	
###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())
