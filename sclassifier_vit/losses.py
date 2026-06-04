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
	
		# 1. Detect one-hot encoded labels [B, C] and convert to class indices [B]
		if targets.ndim == 2 and targets.shape[1] == logits.shape[1]:
			targets = targets.argmax(dim=1)
			
		# 2. Ensure targets are int64, and forcefully reshape them to [B, 1]
		targets = targets.long().view(-1, 1)
	
		# - logits: [B, C], targets: [B] int64
		log_probs = F.log_softmax(logits, dim=1)              # [B, C]
		probs = torch.exp(log_probs)                          # [B, C]
		
		# - pick the prob/log_prob of the target class
		# Since targets is already [B, 1], we don't unsqueeze here
		pt = probs.gather(1, targets).squeeze(1)              # [B]
		###pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)       # [B]
		log_pt = log_probs.gather(1, targets).squeeze(1)      # [B]
		###log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B]
		focal_term = (1.0 - pt).clamp_min(1e-8).pow(self.gamma)      # [B]

		if self.alpha is None:
			alpha_t = 1.0
		elif isinstance(self.alpha, float):
			alpha_t = self.alpha
		else:
			# alpha is tensor [C]
			# We squeeze targets back to [B] so it matches the 1D alpha tensor
			alpha_t = self.alpha.to(logits.device).gather(0, targets.squeeze(1))
			###alpha_t = self.alpha.to(logits.device).gather(0, targets)

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
##    SCORE-ORIENTED LOSS
##########################################
class ScoreOrientedLoss(nn.Module):
	"""
		PyTorch implementation of SOL (Guastavino & Marchetti 2021).
		Defaults to uniform prior + score='tss' ⇒ loss ≈ -TSS (optionally + constant).

		Args:
			- score: one of ['accuracy','precision','recall','specificity','f1_score','tss','csi','hss1','hss2']
			- distribution: 'uniform' or 'cosine'
			- mu, delta: float or list[float] (used only for 'cosine'); length K if multiclass with class-wise params
			- mode: 'average' or 'weighted' (aggregation across one-vs-rest tasks in multiclass)
			- from_logits: if True, apply sigmoid (binary) or softmax (multiclass)
			- add_constant: if True, return `-score + 1`; if False, return `-score`
		Usage:
			# binary: y_true (B,), y_pred (B,) or (B,1)
			# multiclass: y_true (B,) int64 labels; y_pred (B,K)
	"""

	def __init__(
		self,
		score_fn: str = "tss",
		distribution: str = "uniform",
		mu: Union[float, List[float]] = 0.5,
		delta: Union[float, List[float]] = 0.1,
		mode: str = "average",
		#from_logits: bool = True,
		add_constant: bool = False,
	):
		super().__init__()
		
		_VALID_SCORE_FN= ["accuracy", "precision", "recall", "specificity", "f1", "tss", "csi", "hss1", "hss2"]
		
		assert score_fn in _VALID_SCORE_FN, f"Unknown score: {score_fn}"
		assert distribution in ("uniform", "cosine")
		assert mode in ("average", "weighted")
		self.score_fn = score_fn
		self.distribution = distribution
		self.mu = mu
		self.delta = delta
		self.mode = mode
		#self.from_logits = from_logits
		self.add_constant = add_constant


	def _F_uniform(self, p: torch.Tensor) -> torch.Tensor:
		# uniform prior over threshold ⇒ F(p) = p
		return p

	def _F_cosine(self, p: torch.Tensor, mu: float, delta: float) -> torch.Tensor:
		# Raised cosine CDF as in the TF code
		# piecewise: 0 for p < mu-delta; 1 for p > mu+delta; smoothed in between
		out = torch.zeros_like(p)
		left = mu - delta
		right = mu + delta

		# middle region
		mid_mask = (p >= left) & (p <= right)
		out[p > right] = 1.0

		# 0.5*(1 + (p-mu)/delta + 1/pi * sin(pi*(p-mu)/delta))
		pm = (p[mid_mask] - mu) / delta
		out[mid_mask] = 0.5 * (1.0 + pm + (1.0 / math.pi) * torch.sin(math.pi * pm))	
		return out.clamp(0.0, 1.0)

	def _apply_distribution(self, p: torch.Tensor, j: Optional[int] = None) -> torch.Tensor:
		""" Select distribution """
		if self.distribution == "uniform":
			return self._F_uniform(p)
		else:
			# cosine; allow per-class mu/delta if list provided
			if isinstance(self.mu, list) or isinstance(self.mu, tuple):
				mu = float(self.mu[j])
			else:
				mu = float(self.mu)
			if isinstance(self.delta, list) or isinstance(self.delta, tuple):
				delta = float(self.delta[j])
			else:
				delta = float(self.delta)
			return self._F_cosine(p, mu, delta)

	def _expected_confusion(self, y_true, p, j: Optional[int] = None):
		"""
			y_true: (B,) in {0,1}  float or long
			p     : (B,) in [0,1]  probability for positive class
			Returns scalars TP,TN,FP,FN with grad.
		"""
		y = y_true.float()
		# F_unif(p) = p
		# Fp = p
		Fp = self._apply_distribution(p, j=j).clamp(0.0, 1.0)
		
		TN = torch.sum((1.0 - y) * (1.0 - Fp))
		TP = torch.sum(y * Fp)
		FP = torch.sum((1.0 - y) * Fp)
		FN = torch.sum(y * (1.0 - Fp))
		
		return TN, FP, FN, TP

	def _compute_score_from_confusion(self, TN, FP, FN, TP, which):
		""" Compute score from confusion values """
		# all tensors (scalar) with grad
		eps = 1e-12
		which = which.lower()
    
		if which == 'tss':
			# recall + specificity - 1
			#rec = TP / torch.nan_to_num(TP + FN, nan=0.0)      # TP / (TP+FN)
			#spe = TN / torch.nan_to_num(TN + FP, nan=0.0)      # TN / (TN+FP)
			
			rec_den = TP + FN
			spe_den = TN + FP
			rec = torch.where(rec_den > 0, TP / (rec_den + eps), torch.zeros_like(TP))
			spe = torch.where(spe_den > 0, TN / (spe_den + eps), torch.zeros_like(TN))
			return rec + spe - 1.0
			
		elif which == 'accuracy':
			#return (TP + TN) / torch.nan_to_num(TP + TN + FP + FN, nan=0.0)
			den = TP + TN + FP + FN
			return torch.where(den > 0, (TP + TN) / (den + eps), torch.zeros_like(den))

		elif which == 'precision':
			#return TP / torch.nan_to_num(TP + FP, nan=0.0)
			den = TP + FP
			return torch.where(den > 0, TP / (den + eps), torch.zeros_like(den))

		elif which == 'recall':
			#return TP / torch.nan_to_num(TP + FN, nan=0.0)
			den = TP + FN
			return torch.where(den > 0, TP / (den + eps), torch.zeros_like(den))
			
		elif which == 'specificity':
			#return TN / torch.nan_to_num(TN + FP, nan=0.0)
			den = TN + FP
			return torch.where(den > 0, TN / (den + eps), torch.zeros_like(den))
			
		elif which == 'f1':
			prec = TP / torch.nan_to_num(TP + FP, nan=0.0)
			rec  = TP / torch.nan_to_num(TP + FN, nan=0.0)
			return 2 * (prec * rec) / torch.nan_to_num(prec + rec, nan=0.0)
			
		elif which == 'csi':
			return TP / torch.nan_to_num(TP + FP + FN, nan=0.0)
		elif which == 'hss1':
			return (TP - FP) / torch.nan_to_num(TP + FN, nan=0.0)
		elif which == 'hss2':
			num = 2 * (TP * TN - FP * FN)
			den = (TP + FN) * (FN + TN) + (TP + FP) * (TN + FP)
			return num / torch.nan_to_num(den, nan=0.0)
		else:
			raise ValueError(f"Unknown SOL score: {which}")


	def _compute_binary_score(self, logits, labels):
		""" Compute binary class loss """

		# binary: logits (B,1) or (B,)
		if logits.ndim == 2 and logits.shape[-1] == 1:
			logits_bin = logits.squeeze(-1)
		else:
			logits_bin = logits
			
		prob_pos = torch.sigmoid(logits_bin)                 # (B,)
		y_bin    = labels.float().view(-1)                   # (B,)
		TN, FP, FN, TP = self._expected_confusion(y_bin, prob_pos, j=None)
		score = self._compute_score_from_confusion(TN, FP, FN, TP, which=self.score_fn)
		
		return score
		

	def _compute_multiclass_score(self, logits, labels):
		""" Compute multiclass loss """
		
		C = logits.shape[-1]
		
		# multiclass one-vs-rest on softmax probabilities
		probs = torch.softmax(logits, dim=-1)                # (B,C)
		
		#y_idx = labels.view(-1).long()  # (B,)
		
		# ---------------------------------------------------------
		# Detect if labels are already one-hot [B, C]
		# ---------------------------------------------------------
		if labels.ndim == 2 and labels.shape[1] == C:
			# Already one-hot encoded, just ensure it's float
			y_onehot = labels.float()
		else:
			# Labels are class indices [B] or [B, 1], so we build the one-hot
			y_idx = labels.view(-1).long()                       # (B,)
			y_onehot = torch.zeros_like(probs).scatter_(1, y_idx.unsqueeze(1), 1.0)  # (B,C)
		# ---------------------------------------------------------
		
		# build one-hot without breaking grad path
		#y_onehot = torch.zeros_like(probs).scatter_(1, y_idx.unsqueeze(1), 1.0)  # (B,C)
		
		per_class_scores = []
		per_class_weights = []  # for 'weighted' mode: weight by #negatives like the TF ref
		
		for j in range(C):
			p_j = probs[:, j]               # (B,)
			y_j = y_onehot[:, j]            # (B,)
			
			TN, FP, FN, TP = self._expected_confusion(y_j, p_j, j=j)
			s_j = self._compute_score_from_confusion(TN, FP, FN, TP, which=self.score_fn)
			per_class_scores.append(s_j)
			
			# weight by #negatives in batch (like original SOL 'weighted' option)
			n_neg = torch.clamp((y_j.shape[0] - y_j.sum()), min=1.0)
			per_class_weights.append(n_neg)
			
		scores = torch.stack(per_class_scores)               # (C,)
		if self.mode.lower() == 'weighted':
			w = torch.stack(per_class_weights)               # (C,)
			score = (scores * w).sum() / w.sum()
		else:
			score = scores.mean()
			
		return score

	def forward(self, logits, labels):
	
		# - Compute score
		C = logits.shape[-1] if logits.ndim == 2 else 1
		
		if C == 1: # binary
			#print(f"Computing binary score (C={C}, logits.ndim={logits.ndim}, logits.shape[-1]={logits.shape[-1]}) ...")
			score= self._compute_binary_score(logits, labels)
		else:
			#print(f"Computing multiclass score (C={C}, logits.ndim={logits.ndim}, logits.shape[-1]={logits.shape[-1]})) ...")
			score= self._compute_multiclass_score(logits, labels)
			
		# - Compute final loss
		if self.add_constant: # TSS=[-1,1] --> LOSS=-TSS=[-1,1] --> LOSS=[0,2] 
			loss_sol= -score + 1.0
		else:
			loss_sol= -score
			
		#print("loss_sol")
		#print(loss_sol)

		return loss_sol
