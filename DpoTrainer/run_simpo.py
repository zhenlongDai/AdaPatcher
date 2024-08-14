#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import random
import sys

import torch
import transformers
from transformers import set_seed
#from transformers import AutoTokenizer
# from alignment import (
#     #DataArguments,
#     #DPOConfig,
#     #H4ArgumentParser,
#     #ModelArguments,
#     get_checkpoint,
#     #get_datasets,
#     #get_kbit_device_map,
#     #get_peft_config,
#     #get_quantization_config,
#     get_tokenizer,
#     #is_adapter_model,
# )
from utils.Dpo_arguments import DataArguments,H4ArgumentParser, ModelArguments, SimPOConfig

from peft import PeftConfig, PeftModel, LoraConfig
from simpo_trainer import SimPOTrainer
from dataclasses import dataclass, field
from typing import Optional, Literal
import os
from utils.load_data2Prefer import processClass
from utils.arguments import train_config
from datasets import DatasetDict
from peft import get_peft_model
import yaml
from configs.config_utils import generate_peft_config
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer
)
logger = logging.getLogger(__name__)


def get_peft_config(model_args: ModelArguments) -> PeftConfig | None:
    if model_args.use_peft is False:
        return None

    peft_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=model_args.lora_target_modules,
        modules_to_save=model_args.lora_modules_to_save,
    )

    return peft_config

def get_tokenizer(
    model_args: ModelArguments, data_args: DataArguments, auto_set_chat_template: bool = True
) :
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path
        if model_args.tokenizer_name_or_path is None
        else model_args.tokenizer_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if data_args.truncation_side is not None:
        tokenizer.truncation_side = data_args.truncation_side

    # Set reasonable default for models without max length
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 2048


    return tokenizer


    # training hyperparams
    # seed:int = field(default=42, metadata={"help": "seed"})

    # max_length: int = field(
    #     default=2048,
    #     metadata={"help": "the max sentence sequence length."}
    # )   


def main():
    
    
    parser = H4ArgumentParser((ModelArguments, DataArguments, SimPOConfig)) 
    model_args, data_args, training_args = parser.parse()
    #print(training_args)
    base_model = LlamaForCausalLM.from_pretrained(
        "./output_dir/loraWeight/trace_CRFLP/checkpoint-14000"
        #**model_kwargs,
    )
    
    
    input()
   

    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")


    # # Set seed for reproducibility
    set_seed(training_args.seed)

    # input()
    # ###############
    # # Load datasets
    # ###############
    
    rank = int(os.environ["RANK"])
    proc = processClass()
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)
    #tokenizer = LlamaTokenizer.from_pretrained(path)
    train_data_set = proc.get_dataset(training_args,tokenizer, "train", is_test = False,rank = rank)
    eval_dataset_set = proc.get_dataset(training_args,tokenizer,  "dev", is_test = False,rank = rank)
    raw_datasets = DatasetDict({
        "train": train_data_set,
        "dev": eval_dataset_set
    })
   

    # #####################################
    # # Load tokenizer and process datasets
    # #####################################

    # Log a few random samples from the training set:
    for index in random.sample(range(len(raw_datasets["train"])), 3):
        logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
        logger.info(f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['chosen']}")
        logger.info(f"Rejected sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['rejected']}")
    #print(type(raw_datasets))


    # torch_dtype = (
    #     model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    # )


    # model_kwargs = dict(
    #     revision=model_args.model_revision,
    #     trust_remote_code=model_args.trust_remote_code,
    #     use_flash_attention_2=model_args.use_flash_attention_2,
    #     torch_dtype=torch_dtype,
    #     use_cache=False if training_args.gradient_checkpointing else True,
    #     device_map= None,
    #     #quantization_config=quantization_config,
    # )

    #model = model_args.model_name_or_path
    
    # logger.info(f"Loading SFT adapter for {model_args.model_name_or_path=}")
    # peft_config = PeftConfig.from_pretrained(model_args.model_name_or_path)
    # model_kwargs = dict(
    #     revision=model_args.base_model_revision,
    #     trust_remote_code=model_args.trust_remote_code,
    #     use_flash_attention_2=model_args.use_flash_attention_2,
    #     torch_dtype=torch_dtype,
    #     use_cache=False if training_args.gradient_checkpointing else True,
    #     device_map=None,
    #     #quantization_config=quantization_config,
    # )

    print(f" model_args.model_name_or_path = { model_args.model_name_or_path}")
    base_model = LlamaForCausalLM.from_pretrained(
        "./CodeLlama-7b-Instruct-hf"
        #**model_kwargs,
    )
    # input_text = "Hello, world!"
    # input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to('cuda')
    # outputs = base_model.generate(input_ids, max_length=50)
    # print(f"Generated text: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
    # # input()
    # if rank == 0:
    #     for name,param in base_model.named_parameters():
    #             print(f"parameter name:{name}, size: {param.size()}")
    # input()
    # # model = PeftModel.from_pretrained(
    # #     base_model,
    # #     model_args.model_name_or_path,
    # #     revision=model_args.model_revision,
    # # )
    # #peft_config = generate_peft_config(model_args, model_kwargs)
    # model = get_peft_model(base_model, peft_config)
    # # model = AutoModelForCausalLM.from_pretrained(
    # #     peft_config.base_model_name_or_path,
    # # )
    # # peft_config = generate_peft_config(args, model_kwargs)
    # # model = get_peft_model(model, peft_config)


    # # 打印模型的词嵌入层和分词器的词汇表大小以验证
    # if rank == 0:
    #     print(f"Model embedding size: {model.get_input_embeddings().weight.size(0)}")
    #     print(f"Tokenizer vocabulary size: {len(tokenizer)}")
    # model = model.cuda()
    # model.print_trainable_parameters()
    # if rank == 0:
    #     for name,param in model.named_parameters():
    #         print(f"parameter name:{name}, size: {param.size()}")
    # model_kwargs = None

    # ref_model = model
    # ref_model_kwargs = model_kwargs

    # if model_args.use_peft is True:
    #     ref_model = None
    #     ref_model_kwargs = None

    # #########################
    # # Instantiate SimPO trainer
    # #########################
    # trainer = SimPOTrainer(
    #     model=model,
    #     ref_model=ref_model, # pass in to bypass DPO Trainer check for ref model but is not actually used
    #     model_init_kwargs=model_kwargs,
    #     ref_model_init_kwargs=ref_model_kwargs,
    #     args=training_args,
    #     beta=training_args.beta,
    #     train_dataset=raw_datasets["train"],
    #     eval_dataset=raw_datasets["test"],
    #     tokenizer=tokenizer,
    #     max_length=training_args.max_length,
    #     max_prompt_length=training_args.max_prompt_length,
    #     peft_config=get_peft_config(model_args),
    #     loss_type=training_args.loss_type,
    # )
    # input()
    # ###############
    # # Training loop
    # ###############
    # checkpoint = None
    # if training_args.resume_from_checkpoint is not None:
    #     checkpoint = training_args.resume_from_checkpoint
    # elif last_checkpoint is not None:
    #     checkpoint = last_checkpoint
    # train_result = trainer.train(resume_from_checkpoint=checkpoint)
    # metrics = train_result.metrics
    # metrics["train_samples"] = len(raw_datasets["train"])
    # trainer.log_metrics("train", metrics)
    # trainer.save_metrics("train", metrics)
    # trainer.save_state()

    # logger.info("*** Training complete ***")

    # ##################################
    # # Save model and create model card
    # ##################################
    # logger.info("*** Save model ***")
    # trainer.save_model(training_args.output_dir)
    # logger.info(f"Model saved to {training_args.output_dir}")

    # # Save everything else on main process
    # kwargs = {
    #     "finetuned_from": model_args.model_name_or_path,
    #     "dataset": list(data_args.dataset_mixer.keys()),
    #     "dataset_tags": list(data_args.dataset_mixer.keys()),
    #     "tags": ["alignment-handbook"],
    # }
    # if trainer.accelerator.is_main_process:
    #     trainer.create_model_card(**kwargs)
    #     # Restore k,v cache for fast inference
    #     trainer.model.config.use_cache = True
    #     trainer.model.config.save_pretrained(training_args.output_dir)

    # ##########
    # # Evaluate
    # ##########
    # # if training_args.do_eval:
    # #     logger.info("*** Evaluate ***")
    # #     metrics = trainer.evaluate()
    # #     metrics["eval_samples"] = len(raw_datasets["test"])
    # #     trainer.log_metrics("eval", metrics)
    # #     trainer.save_metrics("eval", metrics)

    # # if training_args.push_to_hub is True:
    # #     logger.info("Pushing to hub...")
    # #     trainer.push_to_hub(**kwargs)

    # logger.info("*** Training complete! ***")


if __name__ == "__main__":
    main()
