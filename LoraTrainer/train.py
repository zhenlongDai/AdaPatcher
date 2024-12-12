from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import json
import os
from utils.arguments import train_config,test_config
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
from peft import get_peft_model
from configs.config_utils import generate_peft_config
from LoraTrainer.trainer import ModelTrainer, CustomTrainer
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer
)
import transformers 
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
    #input()
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

    #print("Initial sample from train_dataset:", train_data_set[0])
    #print("Initial batch from data_collator:", train_data_set.collate_batch(train_data_set[0]))

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
        if rank == 0:
            print(f"Model embedding size: {model.get_input_embeddings().weight.size(0)}")
            print(f"Tokenizer vocabulary size: {len(tokenizer)}")
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
            
    model.is_parallelizable = True
    model.model_parallel = True
            
    # build trainer
    #---------------------------------------------------------------------------------

    if args.do_train:
        # model, _, _, _  = deepspeed.initialize(args=args, 
        #                                                           model=model,
		# 						model_parameters=model.parameters(), 
        #                         config=args.speedjson)
        # 创建一个 DataLoader，使用和 Trainer 相同的参数
        # data_loader = DataLoader(train_data_set, batch_size=2, collate_fn=train_data_set.collate_batch)

        # # 手动遍历 DataLoader 看看输出
        # for batch in data_loader:
        #     print("Batch from manual DataLoader:", batch)
        #     break
        # input()
        if args.use_peft == True:
            trainer = ModelTrainer(
                model = model,
                args = args,
                train_dataset = train_data_set,
                eval_dataset = eval_dataset_set,
                tokenizer = tokenizer,
                data_collator= train_data_set.collate_batch,
            )  
        else:
            trainer = CustomTrainer(
                model = model,
                args = args,
                train_dataset = train_data_set,
                eval_dataset = eval_dataset_set,
                tokenizer = tokenizer,
                data_collator= train_data_set.collate_batch,
                compute_metrics=compute_metrics,
                logging_dir = args.logging_dir
                #save_safetensors=False,
                #save_strategy = "steps"
            )  
        # 从上次保存的检查点继续训练
        

        if args.resume == True:
            trainer.train(resume_from_checkpoint=True)    
        else:
            if args.input_dir != "None":
                trainer.train(resume_from_checkpoint= args.input_dir)
            else:    
                train_result = trainer.train()

        # metrics = train_result.metrics
        # trainer.log_metrics("train", metrics)
        # trainer.save_metrics("train", metrics)

  
        # eval_result = trainer.evaluate(eval_dataset=eval_dataset_set, metric_key_prefix="eval_")
        # if rank == 0:
        #     print(eval_result)

if __name__ == "__main__":
    main()