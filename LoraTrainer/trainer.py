import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader,Dataset
from transformers import Trainer, AutoConfig
from transformers import EvalPrediction
from torch.utils.data.distributed import DistributedSampler
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss



def print_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

# def compute_metrics(prediction: EvalPrediction):
#     logits = torch.from_numpy(prediction.predictions)
#     scores = torch.from_numpy(prediction.label_ids)
#     logits_diff = logits.unsqueeze(1) - logits.unsqueeze(2)  # [batch_size, num_sample, num_sample]

#     score_mask_larger = (scores.unsqueeze(1) > scores.unsqueeze(2)) * 1.
#     score_mask_smaller = (scores.unsqueeze(1) < scores.unsqueeze(2)) * 1.
#     score_mask = score_mask_larger - score_mask_smaller
#     pad_mask = (scores >= 0).unsqueeze(1) * 1. * (scores >= 0).unsqueeze(2)

#     # calculate accuracy...
#     pred_compare = (logits_diff.detach() > 0.) * 1.
#     total_mask = (score_mask_larger + score_mask_smaller) * pad_mask
#     correct_compare = (pred_compare == score_mask_larger) * total_mask
    
#     all_acc = correct_compare.sum() / total_mask.sum()
#     first_two_acc =  (correct_compare[:, 0, 1]).sum() / (total_mask[:, 0, 1]).sum() 
    
#     return {"Preference total Acc": all_acc.item(), "First-two Acc": first_two_acc.item()}





def gather_all_with_local_grad(tensor, dim=0):
    local_rank = torch.distributed.get_rank()

    with torch.no_grad():
        all_tensors = [torch.zero_like(tensor) for _ in range(dist.get_world_size())]
        torch.distributed.all_gather(all_tensors, tensor)
    all_tensors[local_rank] = tensor

    return torch.stack(all_tensors, dim=dim)
    

def language_modeling_loss0(logits, labels, eps=1e-7):
    batch_size, seq_length, vocab_size = logits.shape
    #Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    
    return loss

#batch=1 language_part_loss = loss1  else loss1 is large
def language_modeling_loss1(lm_logits, labels, attention_mask, eps=1e-7):
    batch_size, seq_length, vocab_size = lm_logits.shape
    num_ignore_labels = torch.sum(labels[:, 1:] == -100, dim=1).float()
    # 计算逐位置的交叉熵损失
    lm_probs = F.cross_entropy(
        input=lm_logits[:, :-1, :].reshape(-1, vocab_size), 
        target=labels[:, 1:].reshape(-1),
        reduction='none'
    ).view(batch_size, -1)
    # 计算加权损失 一般会计算batch内包括padding位置(attention_mask.shape[-1] seq_length) 所以不一致 attention_mask 1的位置
    loglikeli = (lm_probs * attention_mask[:, 1:].float()).sum(dim=-1) / (attention_mask[:, 1:].float().sum(dim=-1)-num_ignore_labels)#seq_length #(attention_mask[:, 1:].float().sum(dim=-1))
    return loglikeli.mean()
    
    
# llm loss
def language_modeling_loss2(lm_logits, labels, eps= 1e-7): 
    batch_size, seq_length, vocab_size = lm_logits.shape
    
    lm_probs = torch.nn.functional.cross_entropy(
        input=lm_logits[:, :-1,:].reshape(-1, vocab_size), 
        target=labels[:, 1:].reshape(-1),
        reduction='mean'
    )#.view(batch_size, -1)
    return lm_probs

def language_part_loss(lm_logits, labels, mask, debug = False, eps= 1e-7): 
    batch_size, seq_length, vocab_size = lm_logits.shape
    # 计算逐位置的交叉熵损失
    lm_probs = F.cross_entropy(
        input=lm_logits[:, :-1, :].reshape(-1, vocab_size), 
        target=labels[:, 1:].reshape(-1),
        reduction='none'
    ).view(batch_size, -1)
    num_ignore_labels = torch.sum(labels[:, 1:] == -100, dim=1).float()
    total_num_ignore_labels = torch.sum(num_ignore_labels)
    valid_mask = (labels != -100).float()
    if mask is not None:
        valid_mask = valid_mask * mask
    #valid_loss = loss_none[valid_mask]
    #mean_loss = valid_loss.mean()

    # if debug == True:
    #     num_ones = torch.sum(valid_mask)
    #     print(f"num_ones = {num_ones}")

    #     # 找到 loss_mask 中值为 1 的位置
    #     mask_indices = torch.nonzero(valid_mask[:, 1:] == 1)
    #     print(f"Number of 1s in valid_mask: {torch.sum(valid_mask == 1).item()}")
    #     print(f"Indices of 1s in valid_mask:\n{mask_indices}")
        
    #     #打印对应的 lm_probs 值
    #     masked_lm_probs = lm_probs[mask_indices[:, 0], mask_indices[:, 1]]
    #     for idx in range(len(mask_indices)):
    #         print(f"Position: {mask_indices[idx].tolist()}, lm_probs: {masked_lm_probs[idx].item()}")
    
    #计算加权损失 一般会计算batch内包括padding位置(attention_mask.shape[-1] seq_length) 所以不一致 attention_mask 1的位置
    loglikeli = (lm_probs * valid_mask[:, 1:].float()).sum() / (batch_size*(seq_length-1)-total_num_ignore_labels)#seq_length #(attention_mask[:, 1:].float().sum(dim=-1))
    return loglikeli

class ModelTrainer(Trainer):
     
    
    def get_train_dataloader(self) -> DataLoader:
        # 确保父类中的方法被正确调用（如果需要）
        # 如果原来的 Trainer 类中已经有这个方法，你可以用 super() 调用它：
        # dataloader = super()._get_train_dataloader()
        # 这里我们手动构建 DataLoader 以进行调试
        sampler = DistributedSampler(self.train_dataset)
        # 添加调试信息
        print("Creating train DataLoader:")
        print(f" - Batch size: {self.args.train_batch_size}")
        print(f" - Shuffle: {sampler.shuffle}")  # DistributedSampler 自动处理 shuffle
        print(f" - Collate fn: {self.data_collator}")

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,  # 使用 sampler 而不是 shuffle
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,  # 如果有多核处理需求可以设置
        )

        # 检查 DataLoader 是否正确初始化
        for i, batch in enumerate(dataloader):
            print(f"Sample batch {i} from DataLoader: {batch}")
            if i > 0:  # 只打印几个批次来检查
                break

        return dataloader
     
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[List[str]] = None):
        device = model.device
   

        with torch.no_grad():
            loss = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()

        if prediction_loss_only:
            return (loss, None, None)
        
        return (loss)

                
    def compute_loss(self, model, inputs, return_outputs=False):
        device = model.device
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        labels = inputs['labels'].to(device)
        
        #batch_size, sample_num, seq_length = input_ids.shape
        
        #if self.args.debug_mode:
        #    print(f">>> input_ids shape {input_ids.shape}")
          
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels= labels,
        )
        
        #hidden_states = outputs['hidden_states'] # shape [bs, seq_length, dim]
        #lm_loss = outputs['loss']
        #logits = outputs.logits
        loss = outputs.loss #shape [bs, seq_length, dim]
        #print(loss_mask)
        #input()
        if 'loss_mask' in inputs:
            #loss_labels = inputs['loss_labels'].to(device)
            loss_mask = inputs['loss_mask'].to(device)
            logits = outputs.logits
            # 创建目标标签（假设目标标签与输入相同）  
            part_loss = language_part_loss(logits, labels, loss_mask, self.args.debug_mode)
            if self.args.debug_mode:
                print_rank_0(f">>> Language modeling loss before {loss}")
                print_rank_0(f"{self.args.gama}")
            loss =  loss + self.args.gama * part_loss
        
        if self.args.debug_mode:
            print_rank_0(f">>> debug")
            #lm_loss0 = language_modeling_loss0(logits, labels)
            #lm_loss1 = language_modeling_loss1(logits, labels, loss_mask)
            
            print_rank_0(f">>> Language modeling loss {loss}")
            if 'loss_mask' in inputs:
                print_rank_0(f">>> Language modeling part_loss {part_loss}")
            
            #print(torch.distributed.get_rank())
            #print(input_ids.shape)
            #input()
        return (loss) if return_outputs else loss     



class CustomTrainer(Trainer):
     
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[List[str]] = None):
        device = model.device
   

        with torch.no_grad():
            loss = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()

        if prediction_loss_only:
            return (loss, None, None)
        
        return (loss)

                
    def compute_loss(self, model, inputs, return_outputs=False):
        device = model.device
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        #batch_size, sample_num, seq_length = input_ids.shape
        
        #if self.args.debug_mode:
        #    print(f">>> input_ids shape {input_ids.shape}")
    
                
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels= input_ids,
        )

        #hidden_states = outputs['hidden_states'] # shape [bs*r, seq_length, dim]
        
        loss = outputs['loss']
        if self.args.debug_mode:
            print_rank_0(f">>> debug")
            print_rank_0(f">>> Language modeling loss {loss}")
            #print(torch.distributed.get_rank())
            #print(input_ids.shape)
        return (loss) if return_outputs else loss       
