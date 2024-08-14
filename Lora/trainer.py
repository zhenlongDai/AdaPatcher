import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers

from transformers import Trainer, AutoConfig
from transformers import EvalPrediction

from utils import print_rank_0


def compute_metrics(prediction: EvalPrediction):
    logits = torch.from_numpy(prediction.predictions)
    scores = torch.from_numpy(prediction.label_ids)
    logits_diff = logits.unsqueeze(1) - logits.unsqueeze(2)  # [batch_size, num_sample, num_sample]

    score_mask_larger = (scores.unsqueeze(1) > scores.unsqueeze(2)) * 1.
    score_mask_smaller = (scores.unsqueeze(1) < scores.unsqueeze(2)) * 1.
    score_mask = score_mask_larger - score_mask_smaller
    pad_mask = (scores >= 0).unsqueeze(1) * 1. * (scores >= 0).unsqueeze(2)

    # calculate accuracy...
    pred_compare = (logits_diff.detach() > 0.) * 1.
    total_mask = (score_mask_larger + score_mask_smaller) * pad_mask
    correct_compare = (pred_compare == score_mask_larger) * total_mask
    
    all_acc = correct_compare.sum() / total_mask.sum()
    first_two_acc =  (correct_compare[:, 0, 1]).sum() / (total_mask[:, 0, 1]).sum() 
    
    return {"Preference total Acc": all_acc.item(), "First-two Acc": first_two_acc.item()}





def gather_all_with_local_grad(tensor, dim=0):
    local_rank = torch.distributed.get_rank()

    with torch.no_grad():
        all_tensors = [torch.zero_like(tensor) for _ in range(dist.get_world_size())]
        torch.distributed.all_gather(all_tensors, tensor)
    all_tensors[local_rank] = tensor

    return torch.stack(all_tensors, dim=dim)
    

class ModelTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[List[str]] = None):
        device = model.device
        labels = inputs['score'].to(device)

        with torch.no_grad():
            loss, logits = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()

        if prediction_loss_only:
            return (loss, None, None)
        
        return (loss, logits, labels)

                
    def compute_loss(self, model, inputs, return_outputs=False):
        device = model.device
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        batch_size, sample_num, seq_length = input_ids.shape
        
        if self.args.debug_mode:
            print(f">>> input_ids shape {input_ids.shape}")
    
                
        outputs = model(
            input_ids=input_ids.view(-1, seq_length),
            attention_mask=attention_mask.view(-1, seq_length),
            padding_side=self.args.padding_side,
            pooling_type=self.args.pooling_type
        )

        hidden_states = outputs['hidden_states'] # shape [bs*r, seq_length, dim]
        
        #batch_logits = outputs['rm_logits'].view(batch_size, sample_num)
       

        lm_loss = outputs['loss']



        if self.args.debug_mode:
            print_rank_0(f">>> debug")
            print_rank_0(f">>> Language modeling loss {lm_loss}")

        
        return (lm_loss, hidden_states) if return_outputs else lm_loss            
