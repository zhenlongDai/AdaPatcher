from typing import List, Optional

from dataclasses import dataclass, field
from transformers import TrainingArguments

@dataclass
class test_config(TrainingArguments):
    max_length: int = field(
        default=4096,
        metadata={"help": "the max sentence sequence length."}
    )   

@dataclass
class train_config(TrainingArguments):
    half: bool = field(
        default= False, 
        metadata={"help": "FP16 inference"}
    )
    do_sample: bool = field(
        default= False, 
        metadata={"help": "do_sample"}
    )
    num_return_sequences: int = field(
        default = 5,
        metadata={"help": "num_return_sequences"}
    )
    newloss_pattern: bool = field(
        default= False, 
        metadata={"help": "newloss_pattern"}
    )
    gama: float = field( default=25.0, metadata={"help": "gama change loss ratio"})

    prompt_pattern: str = field(
        default= "normal", 
        metadata={"help": "normal/diff/trace"}
    )

    eval_pattern: str = field(
        default= "test", 
        metadata={"help": "eval_pattern: train/dev/test"}
    )
    
    logging_dir: str = field(
        default= "./output_dir/logs", 
        metadata={"help": "remove_unused_columns"}
    )

    remove_unused_columns: bool = field(
        default= False, 
        metadata={"help": "remove_unused_columns"}
    )

    use_peft: bool = field(
        default= True, 
        metadata={"help": "use peft"}
    )
    speedjson: str = field(
        default= "./configs/default_offload_opt_param.json", 
        metadata={"help": "speedjson"}
    )
    resume: bool = field(
        default= False, 
        metadata={"help": "continue_train"}
    )
    peft_method: str = field(
        default= "lora", 
        metadata={"help": "peft method"}
    )
    model_name: str = field(
        default= "loraTestModel", 
        metadata={"help": "peft method"}
    )
    save_optimizer: bool = field(
        default= True, 
        metadata={"help": "use peft"}
    )
    scaler: bool = field(
        default= False, 
        metadata={"help": "use peft"}
    )
    predict: bool = field(
        default= True,
        metadata={"help": "predict."}
    )

    gradient_accumulation_steps: int = field(
        default = 1,
        metadata={"help": "gradient_accumulation_steps"}
    )
    enable_fsdp: bool = field(
        default= True,
        metadata={"help": "enable_fsdp."}
    )

    hidden_size: int = field(
        default=4096,
        metadata={"help": "coadellam-7b config.json hidden_size."}
    )
    
    vocab_size: int = field(
        default=32016,
        metadata={"help": "coadellam-7b config.json vocab_size."}
    )

    model_name_or_path: str = field(
        default="./CodeLlama-7b-Instruct-hf", 
        metadata={"help": "the path to load pretrained model."}
    )

    tokenizer_path: str = field(
        default="./CodeLlama-7b-Instruct-hf", 
        metadata={"help": "the path to load pretrained tokenizer."}
    )
    
    best_epoch: int = field(
        default=0,
        metadata={"help": "eval best epoch."}
    ) 

    # mode generate
    temperature: float = field( default=0.0, metadata={"help": "generate temperature"})
    top_p: float = field( default=0.9, metadata={"help": "generate top_p"})
    max_total_seq_len: int = field(default=3000,metadata={"help": "generate_length."})
    max_generate_length: int = field(default=1000,metadata={"help": "generate_length."})  
    
    # experiment setups
    input_dir: str = field(
        default="None", 
        metadata={"help": "output_dir"}
    )

    output_dir: str = field(
        default="./output_dir/loramodel/", 
        metadata={"help": "output_dir"}
    )
    
    predict_dir: str = field(
        default="./predict_dir", 
        metadata={"help": "predict_dirs"}
    )
    predict_filePath: str = field(
        default="./predict_dir/test/test.json", 
        metadata={"help": "predict_filePath"}
    )
    # tokenizer params
    padding_side: str = field(
        default="right",
        metadata={"help": "the direction for tokenizer to add padding tokens."}
    )

    per_device_test_batch_size: int = field(default= 2,metadata={"help": "per_device_test_batch_size."})

    # data params
    language : str = field(default="Python",metadata={"help": "language data."})   
    
    problem_path: str = field(
        default="./repairDataset/Program_Question_Data/English_Program_Question_StringVersion5.json",
        metadata={"help": "the path to load data."}
    )   
    
    data_path: str = field(
        default="./repairDataset/RepairData-PythonLevel/Dataset/",
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

    # training hyperparams
    seed:int = field(default=42, metadata={"help": "seed"})
    #gamma: float = field(default=0.85, metadata={"help": "model gamma"})
    max_length: int = field(
        default=4096,
        metadata={"help": "the max sentence sequence length."}
    )   


