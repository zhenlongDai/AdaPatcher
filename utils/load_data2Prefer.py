import json
from tqdm import tqdm

from copy import deepcopy

import numpy as np
from utils.arguments import train_config
import torch
from torch.utils.data import Dataset
from transformers import LlamaTokenizer
import transformers
from itertools import chain
from transformers import DataCollator
from tqdm import tqdm
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


def create_token_labels(token_ids, tokenizer, seq_length, special_tokens=['-', '+']):
    # 使用分词器对文本进行分词
    #token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # 初始化一个与Token数目相同的mask列表
    mask = [-100] * seq_length
    #print(f"seq_length = {seq_length}")
    # 遍历每个Token，检查它是否对应于特定字符
    LineBreakTokenID = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('\n'))[-1]
    special_tokenIDs = tokenizer.convert_tokens_to_ids(special_tokens)
    before_Token_id = 0

    for idx, token_id in enumerate(token_ids):
        #print(f"idx = {idx} token_id = {token_id}")
        if idx == 0:
            if any(special_token_id == token_id for special_token_id in special_tokenIDs):
                mask[idx] = token_id
        else:
            if any(special_token_id == token_id for special_token_id in special_tokenIDs) and before_Token_id == LineBreakTokenID:
                mask[idx] = token_id
                #print(f"idx = {idx}")
        before_Token_id = token_id


    return  mask
    
def create_token_mask(token_ids, tokenizer, seq_length, special_tokens=['-', '+']):
    # 使用分词器对文本进行分词
    #token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # 初始化一个与Token数目相同的mask列表
    mask = [0] * seq_length
    #print(f"seq_length = {seq_length}")
    # 遍历每个Token，检查它是否对应于特定字符
    LineBreakTokenID = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('\n'))[-1]
    special_tokenIDs = tokenizer.convert_tokens_to_ids(special_tokens)
    before_Token_id = 0

    for idx, token_id in enumerate(token_ids):
        #print(f"idx = {idx} token_id = {token_id}")
        if idx == 0:
            if any(special_token_id == token_id for special_token_id in special_tokenIDs):
                mask[idx] = 1
        else:
            if any(special_token_id == token_id for special_token_id in special_tokenIDs) and before_Token_id == LineBreakTokenID:
                mask[idx] = 1
                #print(f"idx = {idx}")
        before_Token_id = token_id


    return  mask 
class TextDataset(Dataset):
        def __init__(self, data):
            self.data = data 
        def __getitem__(self, index):
            item = self.data[index]
            return item


        def __len__(self,):
            return len(self.data)
        
            
    
class processClass:
    def __init__(self):
        return 
    def get_instruction(self, problem_content, BuggyCode, CorrectContent, language, is_test = False, prompt_pattern = "normal",
        CRPContent = None, anno_input = None,anno_output=None,trace=None,anno_status = None, actual_output = None):
        if prompt_pattern == "trace_CRFLP":
            message = (
                f"Given a programming question and a corresponding piece of buggy code written in {language},"
                f" please provide code repair proposal for the buggy code."
                f" Use `-` to represent the line that maybe need to be deleted or modified."
            )

            instruction =B_SYS + message + E_SYS \
                + f"Programming question: {problem_content}"\
                + f"Buggy code:\n```\n{BuggyCode}```\n"\
                + f"the test execution information:\ninput:{anno_input}expected output:{anno_output}"
            if actual_output != None and  actual_output == 0:
                actual_output = " no output\n"
            instruction += f"actual output:{actual_output}"

            if anno_status == True:
                instruction += f"Program execution trace:\n```\n{trace}```\n"

            instruction += f"Here is the code repair proposal for the buggy code:\n"
            
            
            text = f"{B_INST} {instruction} {E_INST}"
        elif prompt_pattern == "fixbycrflp": 
            message = (
                f"Given a programming question, a corresponding piece of buggy code written in {language}, and code repair proposal for the buggy code."
                f" The code repair proposal uses `-` to represent the line that maybe need to be deleted or modified."
                f" Please correct the code by modifying the provided buggy code."
            )
            instruction = B_SYS + message + E_SYS \
                + f"Programming question: {problem_content}"\
                + f"Buggy code:\n```\n{BuggyCode}```\n"\
                + f"Code repair proposal for the buggy code:\n```\n{CRPContent}```\n"\
                + f"Here is the corrected code:"
            text = f"{B_INST} {instruction} {E_INST}"

        return text 
    
    def get_path(self, data_path, pattern):
        text = f"{data_path}{pattern}.json"
        return text

    def prepare_data_item(self, language, items, problem_content,tokenizer=None, padding=False, is_test = False, prompt_pattern = "normal", pattern = None, use_predict_crp=False):
        data_list = []
    
        for i in range(0,len(items)):
            new_items = {}
            use_items = {}

            anno_status = items[i]['anno_status'][0]


            anno_input = items[i]['input']
            anno_output=items[i]['expected_output']
            actual_output = items[i]['actual_output']
            trace=items[i]['anno_code'][0]

            if prompt_pattern == "trace_CRFLP": 
                CRP_Content = None
            elif prompt_pattern == "fixbycrflp":
                CRP_Content = items[i]['FL_content']
            buggy_code = items[i]['code1']
            new_items['input'] = self.get_instruction(problem_content[i], buggy_code, None, language, is_test,prompt_pattern,CRP_Content,anno_input,anno_output,trace, anno_status, actual_output) 
            
            text = tokenizer(new_items['input'],return_tensors='pt')
            if is_test is False and text['input_ids'].shape[1] > 1800: continue
            if prompt_pattern == "trace_CRFLP": 
                use_items['text_prompt'] = new_items['input']
                use_items['text_chosen'] = f"\n```\n{items[i]['crp_content']}```\n </s>"
                use_items['text_rejected'] = f"\n```\n{items[i]['crp_content2']}```\n </s>"
            elif prompt_pattern == "fixbycrflp": 
                use_items['text_prompt'] = new_items['input']
                use_items['text_chosen'] = f"\n```\n{items[i]['code_content1']}```\n </s>"
                use_items['text_rejected'] = f"\n```\n{items[i]['code_content2']}```\n </s>"
                    
       
            data_list.append(use_items)
            
            
        return  data_list   

          
    def load_json_data(self,data_path):
        with open(data_path, 'r') as f:
            data_list = json.load(f)
        return data_list

    #获取部分还是全部数据
    def get_data_iter(self,data_list, debug=False, is_test=False):
        if debug:
            data_size = len(data_list)
            if is_test:
                up_data_size = 20000#data_size
            else :
                up_data_size = 20000
            data_list = [data_list[i] for i in range(up_data_size)] #min(int(0.1*data_size), up_data_size
   
        #if (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0):
        #    return tqdm(data_list)
        #else:
        return data_list
        
    #可能根据test和train的不同修改成对应的数据内容 字段名可以一致但内容不同
    def load_dataset(self,language, problem_path, data_path, CRPdata_path = None, tokenizer=None, debug=False, padding=False, batch_size = 1,is_test=False, \
        prompt_pattern="normal",rank = 0, pattern = None, use_predict_crp = False):
        if rank == 0:
            print("loading dataset from: \n   {}".format(data_path))
            print("loading CRPdata from: \n   {}".format(CRPdata_path))
        if data_path[-4:] == 'json':
            data_list = self.load_json_data(data_path)
        problem_list = self.load_json_data(problem_path)


        problem_map = dict()
        for item in problem_list:
            problem_map[item['Pid']] = item['ProblemText']
        
        outputs = []
        data_list = self.get_data_iter(data_list, debug=debug, is_test=is_test)
        data_list_len = len(data_list)
        for i in range(0, len(data_list), batch_size):
            items = data_list[i:min(i+batch_size, data_list_len)]
            problem_list =  [problem_map[item['problem_id']] for item in items]
            new_item = self.prepare_data_item(language, items, problem_content = problem_list,tokenizer=tokenizer, padding=padding, \
                is_test = is_test,prompt_pattern = prompt_pattern, pattern = pattern, use_predict_crp = use_predict_crp)
            
            if new_item is not None:
                outputs.append(new_item)
            
            
        outputs = list(chain.from_iterable(outputs))
        if rank == 0:
            print("finished processing {}  data. in {}".format(len(outputs), data_path))
        return outputs

    def get_dataset(self,args, tokenizer, pattern = "train", is_test = False, rank = 0):    
        all_train_data = []
        data_path = self.get_path(args.data_path, pattern)
        # print(args)
        # print(args.per_device_train_batch_size)
        # input()
        train_data = self.load_dataset(
            language = args.language,
            problem_path=args.problem_path,
            data_path=data_path,
            CRPdata_path=args.CRPdata_path,
            tokenizer=tokenizer, 
            debug=args.debug_mode,
            padding=not args.per_device_train_batch_size == 1,
            batch_size = args.per_device_train_batch_size,
            is_test = is_test,
            prompt_pattern = args.prompt_pattern,
            rank = rank,
            pattern = pattern,
            use_predict_crp = args.use_predict_crp,
        )
        all_train_data.extend(train_data)
        if args.debug_mode:
            print(f">>> check tokenized data:")        
            print(f">>> {all_train_data[0]}")

        #train_set = TextDataset(all_train_data)
        # dataset = Dataset.from_dict({"prompt": [item["prompt"] for item in all_train_data],\
        #                               "chosen": [item["chosen"] for item in all_train_data],\
        #                                "rejected": [item["rejected"] for item in all_train_data] })
        return all_train_data

    
      
if __name__ == "__main__":
    parser = transformers.HfArgumentParser(train_config)
    args = parser.parse_args_into_dataclasses()[0]
    args.debug_mode = True
    #print(args)
    
    tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name_or_path)
    print(tokenizer.padding_side)
    tokenizer.add_special_tokens(
            {
                "pad_token": "<PAD>",
            }
        )
    p = processClass()
    p.get_dataset(args, tokenizer, "train", is_test=False)
    
    
    