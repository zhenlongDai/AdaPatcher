from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import json
import os
from utils.arguments import train_config
import torch
from torch.utils.data import DistributedSampler
from utils.load_data import processClass
from Model.model import LoraCodeLlama
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from utils.train_utils import train, clear_gpu_cache, setup_environ_flags
import torch.distributed as dist
from configs.fsdp import fsdp_config
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
)
from peft import get_peft_model
from configs.config_utils import generate_peft_config

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer
)
import transformers 
#from trainer import ModelTrainer
import deepspeed
deepspeed.ops.op_builder.CPUAdamBuilder().load()
# Trainable parameters of the model
def print_trainable_parameters(model):
    total_params = 0
    for name, param in model.module.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
            print(f"Parameter name: {name}, Shape: {param.shape}")
    print(f"Total Trainable Parameters: {total_params}")


def print_parametersStruct(model):
    for name, module in model.named_modules():
                input()
                print(f"Layer Name: {name} -> Module: {module}")

def main(**kwargs):

    dist.init_process_group("nccl")
    parser = transformers.HfArgumentParser(train_config)
    args = parser.parse_args_into_dataclasses()[0]
    #print(args)
    deepspeed.init_distributed()

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    
    
    # torchrun specific
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)
    #print(args)
    

    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_special_tokens(
            {
                "pad_token": "<PAD>",
            }
        )    
    #tokenizer.pad_token = tokenizer.eos_token
    # load data
    #---------------------------------------------------------------------------------  
    proc = processClass()
    train_data_set = proc.get_dataset(args,tokenizer, "train", is_test = False,rank = rank)
    eval_dataset_set = proc.get_dataset(args,tokenizer,  "dev", is_test = False,rank = rank)

    train_sampler = None
    val_sampler = None
    
    train_sampler = DistributedSampler(
        train_data_set,
        rank=dist.get_rank(),
        num_replicas=dist.get_world_size(),
        shuffle=True, #shuffle
    )
    if args.do_eval:
        val_sampler = DistributedSampler(
            eval_dataset_set,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
        )
    
    
    train_dataloader = DataLoader(train_data_set , batch_size=args.per_device_train_batch_size, collate_fn=train_data_set.collate_batch, num_workers=4,sampler=train_sampler if train_sampler else None)
    eval_dataloader = DataLoader(eval_dataset_set , batch_size=args.per_device_eval_batch_size, collate_fn=train_data_set.collate_batch, num_workers=4,sampler=val_sampler if val_sampler else None)


    # inint model
    #---------------------------------------------------------------------------------
    

    
    if args.use_peft == True:
        model = LlamaForCausalLM.from_pretrained(
            args.model_name_or_path,
        )
        peft_config = generate_peft_config(args, kwargs)
        model = get_peft_model(model, peft_config)
        # 扩展模型的词嵌入层，使其与分词器的词汇表大小一致
        model.resize_token_embeddings(len(tokenizer))

        # 打印模型的词嵌入层和分词器的词汇表大小以验证
        print(f"Model embedding size: {model.get_input_embeddings().weight.size(0)}")
        print(f"Tokenizer vocabulary size: {len(tokenizer)}")
        #model.resize_token_embeddings(model.base_model.model.model.embed_tokens.weight.size(0) + 8)
        model = model.cuda()
        model.print_trainable_parameters()
    else:
            
            model = LlamaForCausalLM.from_pretrained(
                args.model_name_or_path,
            )
            peft_config = generate_peft_config(args, kwargs)
            model = get_peft_model(model, peft_config)
            # 扩展模型的词嵌入层，使其与分词器的词汇表大小一致
            model.resize_token_embeddings(len(tokenizer))
            # 打印模型的词嵌入层和分词器的词汇表大小以验证
            print(f"Model embedding size: {model.get_input_embeddings().weight.size(0)}")
            print(f"Tokenizer vocabulary size: {len(tokenizer)}")
            model = LoraCodeLlama(model).cuda()
            

            
    # on_gpu = all(param.is_cuda for param in model.parameters())
    # if on_gpu:
    #     print("模型在 GPU 上运行。")
    # else:
    #     print("模型在 CPU 上运行。")
    # build trainer
    #---------------------------------------------------------------------------------

    if args.do_train:
        model, _, _, _  = deepspeed.initialize(args=args, 
                                                                  model=model,
								model_parameters=model.parameters(), 
                                config=args.speedjson)
        
   
        results = train(
            model,
            train_dataloader,
            eval_dataloader,
            tokenizer,
            args.gradient_accumulation_steps,
            args,
            fsdp_config if args.enable_fsdp else None,
            local_rank if args.enable_fsdp else None,
            rank if args.enable_fsdp else None,
        )  
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]

        
if __name__ == "__main__":
    main()