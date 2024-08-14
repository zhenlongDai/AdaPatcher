from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import json
import os
from utils.arguments import train_config
import torch
from torch.utils.data import DistributedSampler
from Model.model import LoraCodeLlama
from torch.utils.data import DataLoader
from utils.train_utils import train, clear_gpu_cache, setup_environ_flags
import torch.distributed as dist
from peft import get_peft_model, PeftModel
from configs.config_utils import generate_peft_config
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer
)
from utils.load_data import processClass
import transformers 
import torch.distributed as dist
from tqdm import tqdm
from codeTool.utlis.utils import save_data_to_json
import re
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def load_peft_model(model, peft_model):
    peft_model = PeftModel.from_pretrained(model, peft_model)
    return peft_model

def extract_triple_quotes(text):
    # 匹配被三引号 ''' 包裹的内容
    pattern = r"```(.*?)```"
    
    # 使用 findall 寻找所有匹配的内容
    matches = re.findall(pattern, text, flags=re.DOTALL)
    
    if matches:
        # 如果找到匹配的内容，则返回所有被三引号包裹的部分 0为index
        return matches[0]
    else:
        # 如果没有找到匹配的内容，返回原始文本
        return text
    
def Get_code_content(text):
    text = text.split(E_INST)[1]   
    code_content = extract_triple_quotes(text)
    return code_content

def gather_all_outputs(local_output, world_size):
    # Gather outputs from all ranks
    gathered_outputs = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_outputs, local_output)
    
    # Flatten the list of gathered outputs
    all_outputs = [item for sublist in gathered_outputs for item in sublist]
    return all_outputs

def eval_for_baseline(rank, world_size, args, tokenizer, dataloader):
    
    #model = LlamaForCausalLM.from_pretrained(args.model_name_or_path)
    model = LlamaForCausalLM.from_pretrained(args.output_dir)

    model.resize_token_embeddings(len(tokenizer))
    # 打印模型的词嵌入层和分词器的词汇表大小以验证
    print(f"Model embedding size: {model.get_input_embeddings().weight.size(0)}")
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
    model = model.to(rank).half()  # 将模型转换为半精度
    
    # Initialize DeepSpeed for inference
    # ds_engine = deepspeed.init_inference(
    #     model=model,
    #     mp_size=world_size,
    #     dtype=torch.half,
    #     replace_method='auto',
    #     config=args.speedjson
    # )
    # model = ds_engine.module
    # model = model.to(rank)

    model.eval()
    eval_list = []
    print(f'rank = {rank}')
 
    # 生成文本
    for step, batch in enumerate(tqdm(dataloader,colour="green", desc="predict Epoch", dynamic_ncols=True)): 
        input_ids=batch['input_ids'].to(rank)
        user_id_batch=batch['user_id_batch']
        problem_id_batch=batch['problem_id_batch']
        submission1_id_batch=batch['submission1_id_list']
        batch_size = input_ids.shape[0]
        #print(batch_size)
        #attention_mask=batch['attention_mask'].cuda()
        with torch.no_grad():
            with autocast():
                output_sequences = model.generate(input_ids=input_ids, max_new_tokens= (args.max_length/2), pad_token_id = tokenizer.pad_token_id, use_cache=True)

        for i in range(0, batch_size):
            # 解码生成的文本
            generated_text = tokenizer.decode(output_sequences[i], skip_special_tokens=True)
            #print(generated_text)
            #input()
            user_id = user_id_batch[i]
            problem_id = problem_id_batch[i]
            submission1_id = submission1_id_batch[i]
            code_content = Get_code_content(generated_text)
            item = {"user_id":user_id, "problem_id":problem_id, "submission1_id":submission1_id, "code_content": code_content, "origin_generated_text": generated_text.split(E_INST)[1]}
            #item = item.cpu().numpy().tolist()  # 转换为列表
            eval_list.append(item)
            #eval_list.append(item)
    
    # 同步所有进程
    dist.barrier()

    # 收集所有输出
    all_outputs = gather_all_outputs(eval_list, world_size)
    if rank == 0:
        save_data_to_json(all_outputs, args.predict_filePath)

def main(**kwargs):

    dist.init_process_group("nccl")
    parser = transformers.HfArgumentParser(train_config)
    args = parser.parse_args_into_dataclasses()[0]
    #print(args)
    #input()    
    #deepspeed.init_distributed()
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
    

    # inint model
    #---------------------------------------------------------------------------------
    
    tokenizer = LlamaTokenizer.from_pretrained(args.output_dir, padding_side='left')#, padding_side='left'
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # load data
    #---------------------------------------------------------------------------------  
    proc = processClass()#is_test
    eval_dataset_set = proc.get_dataset(args,tokenizer, pattern = args.eval_pattern , is_test = True,rank = rank)
    val_sampler = None

    if args.do_eval:
        val_sampler = DistributedSampler(
            eval_dataset_set,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
        )
    
    eval_dataloader = DataLoader(eval_dataset_set , batch_size=args.per_device_eval_batch_size, collate_fn=eval_dataset_set.collate_batch, num_workers=4,sampler=val_sampler if val_sampler else None)
    print(f"args.per_device_eval_batch_size = {args.per_device_eval_batch_size}")
    
    # 使用多进程进行评测

    eval_for_baseline(rank, world_size, args,tokenizer, eval_dataloader)

            
   
if __name__ == "__main__":
    main()