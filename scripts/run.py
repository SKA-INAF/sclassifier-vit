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
from sclassifier_vit.trainer import MultiLabelClassTrainer, SingleLabelClassTrainer
from sclassifier_vit.custom_transforms import FlippingTransform, Rotate90Transform
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
	parser.add_argument('-datalist','--datalist', dest='datalist', required=True, type=str, help='Input data json filelist') 
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
	
	# - Model options
	parser.add_argument('--predict', dest='predict', action='store_true', help='Predict model on input data (default=false)')	
	parser.set_defaults(predict=False)
	parser.add_argument('--test', dest='test', action='store_true', help='Run model test on input data (default=false)')	
	parser.set_defaults(test=False)
	parser.add_argument('-modelfile', '--modelfile', dest='modelfile', required=False, type=str, default="google/siglip-so400m-patch14-384", action='store',help='Model pretrained file name or weight path to be loaded {google/siglip-large-patch16-256, google/siglip-base-patch16-256, google/siglip-base-patch16-256-i18n, google/siglip-so400m-patch14-384, google/siglip-base-patch16-224}')
	
	parser.add_argument('-ngpu', '--ngpu', dest='ngpu', required=False, type=int, default=1, action='store',help='Number of gpus used for the run. Needed to compute the global number of training steps (default=1)')	
	parser.add_argument('-nepochs', '--nepochs', dest='nepochs', required=False, type=int, default=1, action='store',help='Number of epochs used in network training (default=100)')	
	#parser.add_argument('-optimizer', '--optimizer', dest='optimizer', required=False, type=str, default='adamw', action='store',help='Optimizer used (default=rmsprop)')
	parser.add_argument('-lr_scheduler', '--lr_scheduler', dest='lr_scheduler', required=False, type=str, default='constant', action='store',help='Learning rate scheduler used {constant, linear, cosine, cosine_with_min_lr} (default=constant)')
	parser.add_argument('-lr', '--lr', dest='lr', required=False, type=float, default=5e-5, action='store',help='Learning rate (default=5e-5)')
	parser.add_argument('-min_lr', '--min_lr', dest='min_lr', required=False, type=float, default=1e-6, action='store',help='Learning rate min used in cosine_with_min_lr (default=1.e-6)')
	parser.add_argument('-warmup_ratio', '--warmup_ratio', dest='warmup_ratio', required=False, type=float, default=0.2, action='store',help='Warmup ratio par (default=0.2)')
	parser.add_argument('-batch_size', '--batch_size', dest='batch_size', required=False, type=int, default=8, action='store',help='Batch size used in training (default=8)')
	
	parser.add_argument('--drop_last', dest='drop_last', action='store_true',help='Drop last incomplete batch (default=false)')	
	parser.set_defaults(drop_last=False)
	
	
	#parser.add_argument('--use_warmup_lr_schedule', dest='use_warmup_lr_schedule', action='store_true',help='Use linear warmup+cos decay schedule to update learning rate (default=false)')	
	#parser.set_defaults(use_warmup_lr_schedule=False)
	#parser.add_argument('-nepochs_warmup', '--nepochs_warmup', dest='nepochs_warmup', required=False, type=int, default=10, action='store',help='Number of epochs used in network training for warmup (default=100)')

	parser.add_argument('-augmenter', '--augmenter', dest='augmenter', required=False, type=str, default='v1', action='store', help='Predefined augmenter to be used (default=v1)')
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
	
	parser.add_argument('--multilabel', dest='multilabel', action='store_true',help='Do multilabel classification (default=false)')	
	parser.set_defaults(multilabel=False)
	parser.add_argument('-label_schema', '--label_schema', dest='label_schema', required=False, type=str, default='morph_tags', action='store',help='Predefined label schema to be used {morph_tags,morph_class} (default=morph_tags)')
	
	# - Run options
	parser.add_argument('-device', '--device', dest='device', required=False, type=str, default="cuda:0", action='store',help='Device identifier')
	parser.add_argument('-runname', '--runname', dest='runname', required=False, type=str, default="llava_1.5_radio", action='store',help='Run name')
	parser.add_argument('--verbose', dest='verbose', action='store_true',help='Enable verbose printout (default=false)')	
	parser.set_defaults(verbose=False)
	
	# - Output options
	parser.add_argument('-outdir','--outdir', dest='outdir', required=False, default="", type=str, help='Output data dir') 
	parser.add_argument('--save_model_every_epoch', dest='save_model_every_epoch', action='store_true', help='Save model every epoch (default=false)')	
	parser.set_defaults(save_model_every_epoch=False)
	parser.add_argument('-max_checkpoints', '--max_checkpoints', dest='max_checkpoints', required=False, type=int, default=1, action='store',help='Max number of saved checkpoints (default=1)')
	parser.add_argument('-outfile','--outfile', dest='outfile', required=False, default="classifier_results.json", type=str, help='Output file with saved inference results') 
	
	args = parser.parse_args()	

	return args		
	
			
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
	datalist= args.datalist
	datalist_cv= args.datalist_cv
	
	# - Run options
	device_choice= args.device
	device = torch.device(device_choice if torch.cuda.is_available() else "cpu")
	
	multilabel= args.multilabel
	run_predict= args.predict
	run_test= args.test
	run_eval_on_start= args.run_eval_on_start
	run_eval_on_step= args.run_eval_on_step
	logging_steps= args.logging_steps
	run_name= args.runname
	verbose= args.verbose
	
	outfile= args.outfile
	
	# - Set model name
	modelname= args.modelfile 

	# - Set hyperparameters
	sigmoid_thr= 0.5
	learning_rate= args.lr
	min_lr= args.min_lr
	lr_scheduler= args.lr_scheduler
	nepochs= args.nepochs
	batch_size= args.batch_size
	warmup_ratio= args.warmup_ratio
	#use_warmup_lr_schedule= args.use_warmup_lr_schedule
	#nepochs_warmup= args.nepochs_warmup
	save_model_every_epoch= args.save_model_every_epoch
	max_checkpoints= args.max_checkpoints
	augmenter= args.augmenter
	skip_first_class= args.skip_first_class
	gradient_accumulation_steps= args.gradient_accumulation_steps
	freeze_backbone= args.freeze_backbone
	max_freeze_layer_id= args.max_freeze_layer_id
	drop_last= args.drop_last
	
	# - Set config options
	label_schema= args.label_schema
	if multilabel:
		id2label, label2id, id2target= get_multi_label_target_maps(label_schema, skip_first_class)
	else:
		id2label, label2id, id2target= get_single_label_target_maps(label_schema)
		
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
	# - Create model
	logger.info("Creating model (name=%s) ..." % (modelname))
	model = AutoModelForImageClassification.from_pretrained(
		modelname, 
		problem_type="multi_label_classification" if multilabel else "single_label_classification", 
		id2label=id2label, 
		label2id=label2id,
		num_labels=num_labels
	)
	model= model.to(device)
	
	print("*** MODEL ***")
	print(model)
	print("")
	
	logger.info("Creating processor ...")
	processor = AutoImageProcessor.from_pretrained(modelname)
 
 	# - Freeze backbone?
	if freeze_backbone:
		logger.info("Freezing base layers ...")
		##for param in model.base_model.parameters():
		for name, param in model.base_model.named_parameters():	
			if name.startswith("vision_model.encoder"):
				layer_index= extract_layer_id(name)
				if max_freeze_layer_id==-1 or (max_freeze_layer_id>=0 and layer_index!=-1 and layer_index<max_freeze_layer_id):
					param.requires_grad = False
		
		logger.info("Print base model info ...")	
		for name, param in model.base_model.named_parameters():
			print(name, param.requires_grad)	
				
		logger.info("Print entire model info ...")
		for name, param in model.named_parameters():
			print(name, param.requires_grad)	
	
 	##################################
	##     DATA TRANSFORMS
	##################################
	# - Create data transforms
	logger.info("Creating data transforms ...")
	#size = processor.size["height"]
	size = (processor.size["height"], processor.size["width"])
	mean = processor.image_mean
	std = processor.image_std

	print("*** Image processor config pars ***")
	#print("size")
	#print(size)
	#print("mean")
	#print(mean)
	#print("std")
	#print(std)
	print("do_resize? ", (processor.do_resize))
	print("size: ", (size))
	print("do_rescale? ", (processor.do_rescale))
	print("rescale_factor: ", (processor.rescale_factor))
	print("do_normalize? ", (processor.do_normalize))
	print("mean: ", (mean))
	print("std: ", (std))
	print("do_convert_rgb? ", (processor.do_convert_rgb))
	
	mean= [0.,0.,0.]
	std= [1.,1.,1.]

	sigma_min= 1.0
	sigma_max= 3.0
	ksize= 3.3 * sigma_max
	kernel_size= int(max(ksize, 5)) # in imgaug kernel_size viene calcolato automaticamente dalla sigma così, ma forse si può semplificare a 3x3
	blur_aug= T.GaussianBlur(kernel_size, sigma=(sigma_min, sigma_max))

	transform_v1 = T.Compose(
		[
			T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
			FlippingTransform(),
			Rotate90Transform(),
			#T.ToTensor(),
			T.Normalize(mean=mean, std=std),
		]
	)
	
	transform_v2 = T.Compose(
		[
			T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
			FlippingTransform(),
			Rotate90Transform(),
			T.RandomApply([blur_aug], p=0.1),
			#T.ToTensor(),
			T.Normalize(mean=mean, std=std),
		]
	)
	
	if augmenter=="v1":
		transform_train= transform_v1
	elif augmenter=="v2":
		transform_train= transform_v2
	else:
		transform_train= transform_v1
	
	transform_val = T.Compose(
		[
			T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
			#T.ToTensor(),
			T.Normalize(mean=mean, std=std),
		]
	)
	
	transform_test = T.Compose(
		[
			T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
			#T.ToTensor(),
			T.Normalize(mean=mean, std=std),
		]
	)
	
	#if run_predict:
	#	transform_test= None

	##################################
	##     DATASET
	##################################
	# - Create dataset
	dataset_cv= None
	dataset= None
	nsamples= 0
	nsamples_cv= 0
	
	# - TEST SET
	if run_predict or run_test:
		if multilabel:
			logger.info("Create dataset for prediction/test (multi-label classification) ...")
			dataset= MultiLabelDataset(
				filename=datalist,
				transform=transform_test,
				load_as_gray=args.grayscale,
				apply_zscale=args.zscale, zscale_contrast=args.zscale_contrast,
				resize=args.resize, resize_size=args.resize_size,
				nclasses=nclasses,
				id2target=id2target
			)
		else:
			logger.info("Create dataset for prediction/test (single-label classification) ...")
			dataset= SingleLabelDataset(
				filename=datalist,
				transform=transform_test,
				load_as_gray=args.grayscale,
				apply_zscale=args.zscale, zscale_contrast=args.zscale_contrast,
				resize=args.resize, resize_size=args.resize_size,
				nclasses=nclasses,
				id2target=id2target
			)
		nsamples= dataset.get_sample_size()
	
	else:
		# - TRAIN SET
		if multilabel:
			logger.info("Create train dataset (multi-label classification) ...")
			dataset= MultiLabelDataset(
				filename=datalist,
				transform=transform_train,
				load_as_gray=args.grayscale,
				apply_zscale=args.zscale, zscale_contrast=args.zscale_contrast,
				resize=args.resize, resize_size=args.resize_size,
				nclasses=nclasses,
				id2target=id2target
			)
		else:
			logger.info("Create train dataset (single-label classification) ...")
			dataset= SingleLabelDataset(
				filename=datalist,
				transform=transform_train,
				load_as_gray=args.grayscale,
				apply_zscale=args.zscale, zscale_contrast=args.zscale_contrast,
				resize=args.resize, resize_size=args.resize_size,
				nclasses=nclasses,
				id2target=id2target
			)
		nsamples= dataset.get_sample_size()
		
		# - VALIDATION SET
		if datalist_cv!="":
			if multilabel:
				logger.info("Create val dataset (multi-label classification) ...")
				dataset_cv= MultiLabelDataset(
					filename=datalist_cv,
					transform=transform_val,
					load_as_gray=args.grayscale,
					apply_zscale=args.zscale, zscale_contrast=args.zscale_contrast,
					resize=args.resize, resize_size=args.resize_size,
					nclasses=nclasses,
					id2target=id2target
				)
			else:
				logger.info("Create val dataset (single-label classification) ...")
				dataset_cv= SingleLabelDataset(
					filename=datalist_cv,
					transform=transform_val,
					load_as_gray=args.grayscale,
					apply_zscale=args.zscale, zscale_contrast=args.zscale_contrast,
					resize=args.resize, resize_size=args.resize_size,
					nclasses=nclasses,
					id2target=id2target
				)
			nsamples_cv= dataset_cv.get_sample_size()
 
	# - Create torch dataset	
	def collate_fn(batch):
		pixel_values = torch.stack([item[0] for item in batch])
		labels = torch.stack([item[1] for item in batch])
		return {"pixel_values": pixel_values, "labels": labels}

	#######################################
	##     SET TRAIN/TEST CONFIG OPTIONS
	#######################################
	# - Set output dir
	output_dir= args.outdir
	if output_dir=="":
		output_dir= os.getcwd()
		
	#logger.debug("output_dir=%s" % (output_dir))	
		
	log_dir= os.path.join(output_dir, "logs/")
	#logger.debug("log_dir=%s" % (log_dir))
	
	# - Set eval strategy
	eval_strategy= "no"
	if dataset_cv is not None:
		if run_eval_on_step:
			eval_strategy= "steps"
		else:
			eval_strategy= "epoch"
	
	# - Set training options
	logger.info("Set model options ...")
	training_opts= transformers.TrainingArguments(
		output_dir=output_dir,
		do_train=True if not run_test else False,
		do_eval=True if not run_test and dataset_cv is not None else False,
		do_predict=True if run_test else False,
		num_train_epochs=nepochs,
		#lr_scheduler_type=scheduler,
		learning_rate=learning_rate,
		warmup_ratio=warmup_ratio,
		#warmup_steps=num_warmup_steps,
		per_device_train_batch_size=batch_size,
		per_device_eval_batch_size=batch_size,
		gradient_accumulation_steps=gradient_accumulation_steps,
		dataloader_drop_last= drop_last,
		#eval_strategy="steps" if run_eval_on_step else "epoch",
		eval_strategy=eval_strategy,
		eval_on_start=run_eval_on_start,
		eval_steps=logging_steps,
		##batch_eval_metrics=False,
		##label_names=label_names,# DO NOT USE (see https://discuss.huggingface.co/t/why-do-i-get-no-validation-loss-and-why-are-metrics-not-calculated/32373)
		save_strategy="epoch" if save_model_every_epoch else "no",
		save_total_limit=max_checkpoints,
		logging_dir = log_dir,
		log_level="debug",
		logging_strategy="steps",
		logging_first_step=True,
		logging_steps=logging_steps,
		logging_nan_inf_filter=False,
		#disable_tqdm=True,
		run_name=run_name,
    report_to="wandb",  # enable logging to W&B
	)
	
	# - Set optimizer
	optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
	#training_opts.set_optimizer(name="adamw_torch", learning_rate=learning_rate)
	
	# - Set scheduler
	#training_opts.set_lr_scheduler(
	#	name=lr_scheduler, 
	#	num_epochs=nepochs,
	#	warmup_ratio=warmup_ratio
	#)
	
	tot_batch_size= args.ngpu * batch_size * gradient_accumulation_steps
	if drop_last:
		n_batches = nsamples // batch_size
	else:
		n_batches= math.ceil(nsamples / batch_size)
                    
	num_update_steps_per_epoch = max(n_batches // gradient_accumulation_steps + int(n_batches % gradient_accumulation_steps > 0), 1)
	
	max_steps = math.ceil(nepochs * num_update_steps_per_epoch)
	
	training_steps= max_steps
	warmup_steps = math.ceil(training_steps * warmup_ratio)
	logger.info("Train pars: nsamples=%d, epochs=%d, batch_size=%d, gradacc=%d, tot_batch_size=%d, n_batches=%d, num_update_steps_per_epoch=%d, max_steps=%d, steps=%d, warmup_steps=%d" % (nsamples, nepochs, batch_size, gradient_accumulation_steps, tot_batch_size, n_batches, num_update_steps_per_epoch, max_steps, training_steps, warmup_steps))
	
	if lr_scheduler=="constant":
		scheduler= transformers.get_constant_schedule(optimizer)
	elif lr_scheduler=="linear":
		scheduler= transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps)
	elif lr_scheduler=="cosine":
		scheduler= transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps)
	elif lr_scheduler=="cosine_with_min_lr":
		scheduler= transformers.get_cosine_with_min_lr_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps, min_lr=min_lr)
	else:
		scheduler= transformers.get_constant_schedule(optimizer)	
	
	##scheduler= transformers.get_constant_schedule(optimizer)
	#training_opts.set_lr_scheduler(name="linear", num_epochs=nepochs)
	#num_warmup_steps= 0
	##if use_warmup_lr_schedule:
	##	num_training_steps=nsamples*nepochs
	##	num_warmup_steps=nsamples*nepochs_warmup
	##	logger.info("Setting cosine schedule with warmup (nsteps=%d, nsteps_warmup=%d" % (num_training_steps, num_warmup_steps))
	
	#	#scheduler= transformers.get_cosine_schedule_with_warmup(
	#	#	optimizer,
	#	#	num_training_steps=num_training_steps,
	#	#	num_warmup_steps=num_warmup_steps
	#	#)
		
	#	#training_opts.set_lr_scheduler(
	#	#	name="cosine", 
	#	#	num_epochs=nepochs,
	#	#	warmup_steps=num_warmup_steps
	#	#)
	
	print("--> training options")
	print(training_opts)
	
	# - Set metrics
	if multilabel:
		compute_metrics_custom= build_multi_label_metrics(label_names)
		#compute_metrics_custom= build_multi_label_metrics(None)
	else:
		compute_metrics_custom= build_single_label_metrics(label_names)
		
	# - Initialize trainer
	if run_test:
		logger.info("Initialize model trainer for prediction task ...")
		if multilabel:
			trainer= MultiLabelClassTrainer(
				num_labels=num_labels,
				model=model,
				args=training_opts,
				compute_metrics=compute_metrics_custom,
				tokenizer=processor,
				data_collator=collate_fn,
				optimizers=(optimizer, scheduler)
			)
		else:
			trainer= SingleLabelClassTrainer(
				num_labels=num_labels,
				model=model,
				args=training_opts,
				compute_metrics=compute_metrics_custom,
				tokenizer=processor,
				data_collator=collate_fn,
				optimizers=(optimizer, scheduler)
			)
	else:
		logger.info("Initialize model trainer for training task ...")
		if multilabel:
			trainer= MultiLabelClassTrainer(
				num_labels=num_labels,
				model=model,
				args=training_opts,
				train_dataset=dataset,
				eval_dataset=dataset_cv,
				compute_metrics=compute_metrics_custom,
				tokenizer=processor,
				data_collator=collate_fn,
				optimizers=(optimizer, scheduler)
			)
		else:
			trainer= SingleLabelClassTrainer(
				num_labels=num_labels,
				model=model,
				args=training_opts,
				train_dataset=dataset,
				eval_dataset=dataset_cv,
				compute_metrics=compute_metrics_custom,
				tokenizer=processor,
				data_collator=collate_fn,
				optimizers=(optimizer, scheduler)
			)
			
			
	#######################################
	##     RUN TEST
	#######################################		
	# - Run predict on test set
	if run_test:
		logger.info("Predict model on input data %s ..." % (datalist))
		predictions, labels, metrics= trainer.predict(dataset, metric_key_prefix="predict")
		
		print("type(predictions)")
		print(type(predictions))
		print(predictions)
		
		print("type(labels)")
		print(type(labels))
		print(labels)
		
		print("prediction metrics")
		print(metrics) 
		
		trainer.log_metrics("predict", metrics)
		trainer.save_metrics("predict", metrics)	
	
	################################
	##    RUN PREDICT
	################################
	# - Run test
	elif run_predict:
		logger.info("Running model inference on input data %s ..." % (datalist))
		device = torch.device(device_choice if torch.cuda.is_available() else "cpu")
		
		inference_results= {"data": []}
		
		for i in range(nsamples):
			# - Retrieve image info
			image_info= dataset.load_image_info(i)
			sname= image_info["sname"]
			
			# - Load image & extract embeddings
			#image= dataset.load_pil_image(i)
			#pixel_values= processor(image, return_tensors="pt").pixel_values.to(device)
			
			image_tensor= dataset.load_tensor(i)
			image_tensor= image_tensor.unsqueeze(0).to(device)
 
			with torch.no_grad():
				outputs = model(image_tensor)
				logits = outputs.logits
				
				#outputs_v2 = model(pixel_values)
				#logits_v2 = outputs_v2.logits
				
  		# - Compute predicted labels & probs
			if multilabel:
				sigmoid = torch.nn.Sigmoid()
				probs = sigmoid(logits.squeeze().cpu()).numpy()
				#probs_v2 = sigmoid(logits_v2.squeeze().cpu()).numpy()
				predictions = np.zeros(probs.shape)
				predictions[np.where(probs >= sigmoid_thr)] = 1 # turn predicted id's into actual label names
				
				predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
				predicted_probs = [float(probs[idx]) for idx, label in enumerate(predictions) if label == 1.0]
				if not predicted_labels and skip_first_class:
					#print("max(probs)")
					#print(max(probs))
					min_prob= 1 - max(probs)
					predicted_probs= [min_prob]
					predicted_labels= ["BACKGROUND"]	
					
				# - Fill prediction results in summary dict
				image_info["label_pred"]= list(predicted_labels)
				image_info["prob_pred"]= list(predicted_probs) 
				
				if verbose:
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
					
				if verbose:
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
			
			inference_results["data"].append(image_info)
			
		# - Save json file
		logger.info("Saving inference results with prediction info to file %s ..." % (outfile))
		with open(outfile, 'w') as fp:
			json.dump(inference_results, fp)
	
	################################
	##    TRAIN
	################################
	# - Run model train
	else:
		logger.info("Run model training ...")
		device = torch.device(device_choice if torch.cuda.is_available() else "cpu")
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
		if dataset_cv is not None:
			logger.info("Running model evaluation ...")
			metrics = trainer.evaluate()
			trainer.log_metrics("eval", metrics)
			trainer.save_metrics("eval", metrics)
			print("eval metrics")
			print(metrics) 
	
	return 0
	
###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())
