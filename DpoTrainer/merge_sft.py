# 0. imports
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed
import random
from trl import DPOConfig, DPOTrainer
from utils.load_data2Prefer import processClass
from datasets import Dataset
import pandas as pd
from transformers import TrainingArguments
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer
)
from peft import PeftModel
# Define and parse arguments.
@dataclass
class ScriptArguments(TrainingArguments):
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})
    dpop_lambda: Optional[float] = field(default=50.0, metadata={"help": "the lambda parameter for DPOP loss"})
    # training parameters
    model_name_or_path: Optional[str] = field(
        default="../sft/results/final_checkpoint",
        metadata={"help": "the location of the SFT model name or path"},
    )
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=1, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})

    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"} #4
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    gradient_checkpointing_use_reentrant: Optional[bool] = field(
        default=False, metadata={"help": "whether to use reentrant for gradient checkpointing"}
    )

    lora_alpha: Optional[float] = field(default=8, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.0, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(default=2000, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=2048, metadata={"help": "the maximum sequence length"})
    num_train_epochs: Optional[int] = field(default=5, metadata={"help": "num_train_epochs"})
    #max_steps: Optional[int] = field(default=10000, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=5, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=400, metadata={"help": "the evaluation frequency"})

    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "whether to load the model in 4bit"})
    model_dtype: Optional[str] = field(
        default="float16", metadata={"help": "model_dtype[float16, bfloat16, float] for loading."}
    )
    run_name: Optional[str] = field(default="Dpo_codeLlama2", metadata={"help": "run name for Wandb"})
    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    seed: Optional[int] = field(
        default=0, metadata={"help": "Random seed that will be set at the beginning of training."}
    )

    #new
    language : str = field(default="Python",metadata={"help": "language data."})   
    
    problem_path: str = field(
        default="./repairDataset/Program_Question_Data/English_Program_Question_StringVersion5.json",
        metadata={"help": "the path to load data."}
    )   
    
    data_path: str = field(
        default="./repairDataset/RepairData-PythonLevel/PreferDataset/",
        metadata={"help": "train datasets paths."}
    )
    use_predict_crp: bool = field(
        default=False,
        metadata={"help": "whether use the crp by prediction."}
    )

    CRPdata_path: str = field(
        default="None",
        metadata={"help": "In second stage, fix needs CRP result"}
    )
    debug_mode: bool = field(
        default=False,
        metadata={"help": "whether use the debug mode."}
    )
    
    prompt_pattern: str = field(
        default= "trace_CRFLP", 
        metadata={"help": "normal/diff/trace"}
    )

    eval_pattern: str = field(
        default= "train", 
        metadata={"help": "eval_pattern: train/dev/test"}
    )

    resume: bool = field(
        default= False, 
        metadata={"help": "continue_train"}
    )

    output_name: str = field(
        default= "./output_dir/fix_codeLlama", 
        metadata={"help": "sft name"}
    )


def get_stack_exchange_paired(
    data_dir: str = "data/rl",
    sanity_check: bool = False,
    cache_dir: Optional[str] = None,
    num_proc=24,
) -> Dataset:
    """Load the stack-exchange-paired dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts are structured as follows:
      "Question: " + <prompt> + "\n\nAnswer: "
    """
    dataset = load_dataset(
        "lvwerra/stack-exchange-paired",
        split="train",
        cache_dir=cache_dir,
        data_dir=data_dir,
        verification_mode="no_checks",
    )
    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            "prompt": ["Question: " + question + "\n\nAnswer: " for question in samples["question"]],
            "chosen": samples["chosen"],
            "rejected": samples["rejected"],
        }

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

def get_stack_exchange_paired2(
    data_list,
    sanity_check: bool = False,
    num_proc=24,
) -> Dataset:
    """Load the stack-exchange-paired dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts are structured as follows:
      "Question: " + <prompt> + "\n\nAnswer: "
    """
    df = pd.DataFrame(data_list)

    # 将 DataFrame 转换为 Dataset
    dataset = Dataset.from_pandas(df)
    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 50)))

    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            "prompt": [prompt for prompt in samples["text_prompt"]],
            "chosen": samples["text_chosen"],
            "rejected": samples["text_rejected"],
        }

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    set_seed(script_args.seed)

    # 1. load a pretrained model
    torch_dtype = torch.float
    if script_args.model_dtype == "float16":
        torch_dtype = torch.float16
    elif script_args.model_dtype == "bfloat16":
        torch_dtype = torch.bfloat16

    model = LlamaForCausalLM.from_pretrained(
        "./CodeLlama-7b-Instruct-hf", return_dict=True, torch_dtype=torch.bfloat16
        #script_args.model_name_or_path,
        #low_cpu_mem_usage=True,
        #torch_dtype=torch_dtype,
        #load_in_4bit=script_args.load_in_4bit,
        #device_map={"": Accelerator().local_process_index},
    )
    
    # for name,param in model.named_parameters():
    #     print(f"parameter name:{name}, size: {param.size()}")
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
    # input()
    model = PeftModel.from_pretrained(model, script_args.model_name_or_path)
    tokenizer = LlamaTokenizer.from_pretrained(script_args.model_name_or_path)
    model.print_trainable_parameters()
    model.eval()
    model = model.merge_and_unload()
    model.save_pretrained(f"{script_args.output_name}")
    tokenizer.save_pretrained(f"{script_args.output_name}")
    
   