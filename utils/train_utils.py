# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import time
import yaml
from contextlib import nullcontext
from pathlib import Path
from pkg_resources import packaging
from torch import nn
import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tqdm import tqdm
from transformers import LlamaTokenizer
from utils.memory_utils import MemoryTrace
from utils.checkpoint_utils import save_model_checkpoint, save_model_and_optimizer_sharded, save_optimizer_checkpoint

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig
)
from transformers import EvalPrediction
import functools
from transformers.models.llama.modeling_llama import LlamaDecoderLayer,LlamaRMSNorm,LlamaForCausalLM
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    _module_wrap_policy,
)
import numpy as np

fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

def print_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

def compute_metrics(prediction: EvalPrediction):
    logits, labels = prediction
    predictions = np.argmax(logits, axis=-1)
    accuracy = np.mean(predictions == labels)
    return {'accuracy': accuracy, 'loss': np.mean(logits)}



def load_model_checkpoint(model, rank,cfg):
    if rank != 0:
        return
    
    model_checkpoint_dir = os.path.join(cfg.output_dir, f'model.pt')
    print(f"start model checkpoint loaded in {model_checkpoint_dir}")
    model_checkpoint = torch.load(model_checkpoint_dir)
    model.load_state_dict(model_checkpoint)
    print(f"model checkpoint loaded in {model_checkpoint_dir}")
    return model
 
 

   
def train(model, train_dataloader,eval_dataloader, tokenizer, gradient_accumulation_steps, train_config, fsdp_config=None, local_rank=None, rank=None):
    """
    Trains the model on the given dataloader
    """
  
   
    
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])

    train_prep = []
    train_loss = []
    val_prep = []
    val_loss =[]
    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    stop_cont = 2
    stop_step = 0
    if train_config.resume:
            _, client_state = model.load_checkpoint(train_config.output_dir, tag=None)
            start_epoch = client_state['epoch'] + 1 if client_state else 0
    else:
        start_epoch = 0

    for epoch in range(start_epoch, int(train_config.num_train_epochs)):
        if stop_step >= stop_cont:
            break
        epoch_start_time = time.perf_counter()
        #with MemoryTrace() as memtrace:
        model.train()
        total_loss = 0.0
        total_length = len(train_dataloader)//gradient_accumulation_steps
        pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)
        for step, batch in enumerate(train_dataloader):

            
            input_ids = batch['input_ids'].to(local_rank)
            attention_mask = batch['attention_mask'].to(local_rank)
            # print(input_ids.shape)
            # print(attention_mask.shape)
            # print(input_ids)
            # print(attention_mask)
            # input()
            if train_config.use_peft == True:
                result = model(input_ids,attention_mask,labels = input_ids)
                loss = result.loss
                
            else:

                result = model(input_ids,attention_mask, past_key_values = None)
                loss = result["loss"]
            
            total_loss += loss.detach().float()
                        
            model.backward(loss)  
            model.step() 
        
            pbar.update(1)
            pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_train_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")
        
        torch.cuda.empty_cache()
        pbar.close()
            
        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)    
        # Reducing total_loss across all devices if there's more than one CUDA device
        if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            
        train_epoch_loss = total_loss / len(train_dataloader)
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss/world_size
        train_perplexity = torch.exp(train_epoch_loss)
        train_prep.append(train_perplexity)
        train_loss.append(train_epoch_loss)
        if rank==0:
            print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
          
        if train_config.do_eval:
            eval_ppl, eval_epoch_loss = evaluation(model, train_config, eval_dataloader, tokenizer, local_rank)
            checkpoint_start_time = time.perf_counter()
            if eval_epoch_loss < best_val_loss:
 
                if train_config.use_peft == True:
                    if train_config.enable_fsdp:
                        if rank==0:
                            print(f"we are about to save the PEFT modules")
                    else:
                        print(f"we are about to save the PEFT modules")
                    model.save_pretrained(train_config.output_dir+"{epoch}-{step}/")  
                    if train_config.enable_fsdp:
                        if rank==0: 
                            print(f"PEFT modules are saved in {train_config.output_dir} directory")
                    else:
                        print(f"PEFT modules are saved in {train_config.output_dir} directory")

                else:

                    client_state = {'epoch': epoch, 'step': step}
                    model.save_checkpoint(train_config.output_dir, client_state=client_state)
           
            if eval_epoch_loss >= best_val_loss:
                stop_step += 1
                if stop_step == stop_cont:
                    break
                
            checkpoint_end_time = time.perf_counter() - checkpoint_start_time
            checkpoint_times.append(checkpoint_end_time)
            if eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                if train_config.enable_fsdp:
                    if rank==0:
                        print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
                else:
                    print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
            val_loss.append(best_val_loss)
            val_prep.append(eval_ppl)
        
    avg_epoch_time = sum(epoch_times)/ len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep)/len(train_prep)
    avg_train_loss = sum(train_loss)/len(train_loss)
    if train_config.do_eval:
        avg_eval_prep = sum(val_prep)/len(val_prep) 
        avg_eval_loss = sum(val_loss)/len(val_loss) 

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    
    if train_config.do_eval:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    return results

def evaluation(model,train_config, eval_dataloader, tokenizer, local_rank):
    """
    Evaluates the model on the given dataloader
    
    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions
    
    Returns: eval_ppl, eval_epoch_loss
    """
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])  
    model.eval()
  
    eval_loss = 0.0  # Initialize evaluation loss
    
    with MemoryTrace() as memtrace:
        for step, batch in enumerate(tqdm(eval_dataloader,colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
            if train_config.enable_fsdp:
                input_ids = batch['input_ids'].to(local_rank)
                attention_mask = batch['attention_mask'].to(local_rank)
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss 
                if train_config.use_peft == True:
                    result = model(input_ids,attention_mask,labels = input_ids)
                    loss = result.loss
                
                else:
                    result = model(input_ids,attention_mask, past_key_values = None)
                    loss = result["loss"]
                eval_loss += loss.detach().float()

    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
    
    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    if train_config.enable_fsdp:
        eval_epoch_loss = eval_epoch_loss/world_size
    eval_ppl = torch.exp(eval_epoch_loss)
    
    # Print evaluation metrics
    if train_config.enable_fsdp:
        if local_rank==0:
            print(f" {eval_ppl=} {eval_epoch_loss=}")
    return eval_ppl, eval_epoch_loss


def freeze_transformer_layers(model, num_layer):
   for i, layer in enumerate(model.model.model.layers):
            if i < num_layer:
                for param in layer.parameters():
                    param.requires_grad = False
                    
def setup():
    """Initialize the process group for distributed training"""
    dist.init_process_group("nccl")


def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # This flag will help with CUDA memory fragmentations that can lead into OOM in some cases.
    # Note this is only availble in PyTorch Nighlies (as of July 30 2023)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True' 
    if rank == 0:
        print(f"--> Running with torch dist debug set to detail")


def cleanup():
    """Clean up the process group after training"""
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print(f"Clearing GPU cache for all ranks")
    torch.cuda.empty_cache()


def get_parameter_dtypes(model):
    """Get the data types of model parameters"""
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes



