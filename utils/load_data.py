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
        def __init__(self, data,tokenizer, test, newloss_pattern = False):
            self.data = data 
            self.tokenizer = tokenizer
            self.test = test
            self.newloss_pattern = newloss_pattern
        def __getitem__(self, index):
            
            item = self.data[index]
            #print("Dataset item:", item)  # 打印获取的数据项
            return item
            #return 

        def __len__(self,):
            return len(self.data)
        
        def collate_batch(self, batch):
            #print("@@@@ Batch before collate:", batch)  # 打印原始批次数据

            input_ids_batch = []
            attention_mask_batch = []
            user_id_batch = []
            problem_id_batch = []
            instruction_list = []
            submission1_id_batch = []
            loss_labels_batch = []
            loss_mask_batch = []
            #print(batch)
            
            for item in batch:
                user_id_batch.append(item['user_id'])
                problem_id_batch.append(item['problem_id'])
                submission1_id_batch.append(item['submission1_id'])
                instruction_list.append(item['input'])
                
            encoded_inputs = self.tokenizer(instruction_list,padding=True,return_tensors='pt')
            input_ids_batch = encoded_inputs["input_ids"]
            labels_batch = input_ids_batch.clone()
            attention_mask_batch = encoded_inputs["attention_mask"]
            #tokens_batch =self.tokenizer.convert_ids_to_tokens(input_ids_batch[0])

            batch_size, seq_length = attention_mask_batch.shape
            if self.newloss_pattern == True:
                for input_ids in input_ids_batch:
                    loss_mask = create_token_mask(input_ids, self.tokenizer, seq_length)
                    #loss_labels = create_token_labels(input_ids, self.tokenizer, seq_length)
                    #loss_labels_batch.append(loss_labels)
                    loss_mask_batch.append(loss_mask)
                #loss_labels_batch = torch.Tensor(loss_labels_batch).long()
                loss_mask_batch = torch.Tensor(loss_mask_batch).long()

            if self.test == False:
                # Locate E_INST and modify attention_mask
                E_INST_token_id = self.tokenizer.encode(E_INST, add_special_tokens=False)[2]  # Assuming E_INST is a single token
                input_ids = input_ids_batch[0]

                for i, input_ids in enumerate(input_ids_batch):
                    try:
                        # Find the position of E_INST token
                        end_idx = (input_ids == E_INST_token_id).nonzero(as_tuple=True)[0][-1].item() #
 
                        # Set attention_mask to -100 before E_INST (inclusive)
                        labels_batch[i, :end_idx + 2] = -100
                    except IndexError:
                        # Handle case where E_INST is not found
                        print(f"E_INST not found in input {i}")
                if self.newloss_pattern == True:
                    for i, input_ids in enumerate(input_ids_batch):
                        try:
                            # Find the position of E_INST token
                            end_idx = (input_ids == E_INST_token_id).nonzero(as_tuple=True)[0][-1].item()
                            # Set attention_mask to -100 before E_INST (inclusive)
                            #loss_labels_batch[i, :end_idx + 2] = -100
                            loss_mask_batch[i, :end_idx + 2] = 0
                        except IndexError:
                            # Handle case where E_INST is not found
                            print(f"E_INST not found in input {i}")
            if self.newloss_pattern == True:
                return {
                    "user_id_batch": user_id_batch,
                    "problem_id_batch": problem_id_batch,
                    "submission1_id_list": submission1_id_batch,
                    "input_ids": torch.Tensor(input_ids_batch).long(),
                    "labels": torch.Tensor(labels_batch).long(),
                    #"loss_labels": torch.Tensor(loss_labels_batch).long(),
                    "attention_mask": torch.Tensor(attention_mask_batch).float(),
                    "loss_mask": torch.Tensor(loss_mask_batch).float()
                }
            else:
                return {
                    "user_id_batch": user_id_batch,
                    "problem_id_batch": problem_id_batch,
                    "submission1_id_list": submission1_id_batch,
                    "input_ids": torch.Tensor(input_ids_batch).long(),
                    "labels": torch.Tensor(labels_batch).long(),
                    "attention_mask": torch.Tensor(attention_mask_batch).float(),
                    #"loss_labels": torch.Tensor(loss_labels_batch).long(),
                    #"loss_mask": torch.Tensor(loss_mask_batch).float()
                }
            
    
    
class processClass:
    def __init__(self):
        return 
    def get_instruction(self, problem_content, BuggyCode, CorrectContent, language, is_test = False, prompt_pattern = "normal",
        CRPContent = None, anno_input = None,anno_output=None,trace=None,anno_status = None, actual_output = None):
        if prompt_pattern == "normal":
            instruction =B_SYS + f"Given a programming question and a corresponding piece of buggy code written in {language}, please correct the code by modifying the provided buggy code."+ E_SYS \
                + f"Programming question: {problem_content}"\
                + f"Buggy code:\n```\n{BuggyCode}```\n" + E_SYS
            if is_test:
                text = f"{B_INST} {instruction} {E_INST}"
            else:
                text = f"{B_INST} {instruction} {E_INST} Here is the corrected code:\n```\n{CorrectContent}```\n </s>"
        elif prompt_pattern == "diff":
            message = (

                f"Given a programming question and a corresponding piece of buggy code written in {language},"
                f" use the output format of the ```git diff``` command to correct the buggy code by showing the differences between the corrected and buggy code,"
                f" rather than providing the corrected code directly."
                f" This output shows the differences between the two files,"
                f" with the `+` symbol indicating the lines that were added in the corrected code,"
                f" and the `-` symbol indicating the lines that were removed in the corrected code."
            )
            instruction =B_SYS + message + E_SYS \
                + f"Programming question: {problem_content}"\
                + f"Buggy code:\n```\n{BuggyCode}```\n"\
                + f"Here is the output of the ```git diff``` command: \n"
            if is_test:
                text = f"{B_INST} {instruction} {E_INST}"
            else:
                text = f"{B_INST} {instruction} {E_INST}\n```\n{CorrectContent}```\n </s>"
        elif "trace" in prompt_pattern:
            if "normal" in prompt_pattern:
                message = (
                    f"Given a programming question, a corresponding piece of buggy code written in {language}, and the test execution information for the buggy code,"
                    f" please correct the code by modifying the provided buggy code."
                )
            elif "CRP" in prompt_pattern:
                message = (
                    f"Given a programming question and a corresponding piece of buggy code written in {language},"
                    f" please provide code repair proposal for the buggy code."
                    f" Use `-` to represent the line that need to be deleted or modified;"
                    f" use `+` to indicate the area that need to be added or modified, with no limit on the number of lines."
                )
            elif "CRFLP" in prompt_pattern:
                message = (
                    f"Given a programming question and a corresponding piece of buggy code written in {language},"
                    f" please provide code repair proposal for the buggy code."
                    f" Use `-` to represent the line that maybe need to be deleted or modified."
                )

            
            instruction =B_SYS + message + E_SYS \
                + f"Programming question: {problem_content}"\
                + f"Buggy code:\n```\n{BuggyCode}```\n"\
                + f"the test execution information:\ninput:{anno_input}\nexpected output:{anno_output}\n"
            if actual_output != None and  actual_output == 0:
                actual_output = " no output\n"
            instruction += f"actual output:{actual_output}\n"

            if anno_status == True:
                instruction += f"Program execution trace:\n```\n{trace}```\n"

            if "normal" in prompt_pattern:
                if is_test:
                    text = f"{B_INST} {instruction} {E_INST}"
                else:
                    text = f"{B_INST} {instruction} {E_INST} Here is the corrected code:\n```\n{CorrectContent}```\n </s>"

            elif "CRP" in prompt_pattern or "CRFLP" in prompt_pattern:
                instruction += f"Here is the code repair proposal for the buggy code:\n"
                if is_test:
                    text = f"{B_INST} {instruction} {E_INST}"
                else:
                    text = f"{B_INST} {instruction} {E_INST}\n```\n{CRPContent}```\n </s>"

        elif "CRP" == prompt_pattern:
            message = (
                f"Given a programming question and a corresponding piece of buggy code written in {language},"
                f" please provide code repair proposal for the buggy code."
                f" Use `-` to represent the line that need to be deleted or modified;"
                f" use `+` to indicate the area that need to be added or modified, with no limit on the number of lines."
            )
            instruction =B_SYS + message + E_SYS \
                + f"Programming question: {problem_content}"\
                + f"Buggy code:\n```\n{BuggyCode}```\n"\
                +f"Here is the code repair proposal for the buggy code:\n"
            if is_test:
                text = f"{B_INST} {instruction} {E_INST}"
            else:
                text = f"{B_INST} {instruction} {E_INST}\n```\n{CRPContent}```\n </s>"

        elif "CRFLP" == prompt_pattern:
            message = (
                    f"Given a programming question and a corresponding piece of buggy code written in {language},"
                    f" please provide code repair proposal for the buggy code."
                    f" Use `-` to represent the line that maybe need to be deleted or modified."
                )
            instruction =B_SYS + message + E_SYS \
                + f"Programming question: {problem_content}"\
                + f"Buggy code:\n```\n{BuggyCode}```\n"\
                +f"Here is the code repair proposal for the buggy code:\n"
            if is_test:
                text = f"{B_INST} {instruction} {E_INST}"
            else:
                text = f"{B_INST} {instruction} {E_INST}\n```\n{CRPContent}```\n </s>"

        elif prompt_pattern == "fixbycrp":
            message = (
                f"Given a programming question, a corresponding piece of buggy code written in {language}, and code repair proposal for the buggy code."
                f" The code repair proposal uses `-` to represent the line that maybe need to be deleted or modified:"
                f" use `+` to indicate the area that maybe need to be added or modified, with no limit on the number of lines."
                f" Please correct the code by modifying the provided buggy code."
            )
            instruction = B_SYS + message + E_SYS \
                + f"Programming question: {problem_content}"\
                + f"Buggy code:\n```\n{BuggyCode}```\n"\
                + f"Code repair proposal for the buggy code:\n```\n{CRPContent}```\n"\
                + f"Here is the corrected code:"
            if is_test:
                text = f"{B_INST} {instruction} {E_INST}"
            else:
                text = f"{B_INST} {instruction} {E_INST}\n```\n{CorrectContent}```\n </s>"
                #text = f"{B_INST}abc{E_INST}\n```\n\n+\n-abced```\n </s>"
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
            if is_test:
                text = f"{B_INST} {instruction} {E_INST}"
            else:
                text = f"{B_INST} {instruction} {E_INST}\n```\n{CorrectContent}```\n </s>"

        #print(text)
        #input()
        return text 
    
    def get_path(self, data_path, pattern):
        text = f"{data_path}{pattern}.json"
        return text

    def prepare_data_item(self, language, items, problem_content,tokenizer=None, padding=False, is_test = False, prompt_pattern = "normal", pattern = None, use_predict_crp=False):
        data_list = []
    
        for i in range(0,len(items)):
            new_items = {}
            new_items['code1'] = items[i]['code1']
            new_items['user_id'] = items[i]['user_id']
            
            new_items['problem_id'] = items[i]['problem_id']
            new_items['submission1_id'] = items[i]['submission1_id']

            if prompt_pattern == "normal":
                new_items['code2'] = items[i]['code2']
                new_items['input'] = self.get_instruction(problem_content[i], new_items['code1'], new_items['code2'], language, is_test,prompt_pattern) 
                text = tokenizer(new_items['input'],return_tensors='pt')
            elif prompt_pattern == "diff":
                new_items['code2'] = items[i]['diff_content']
                new_items['input'] = self.get_instruction(problem_content[i], new_items['code1'], new_items['code2'], language, is_test,prompt_pattern) 
                text = tokenizer(new_items['input'],return_tensors='pt')
                if is_test is False and text['input_ids'].shape[1] > 2048: continue #训练数据过长不要了 
            elif prompt_pattern == "fixbycrp" or prompt_pattern == "fixbycrflp":
                new_items['code2'] = items[i]['code2']

                if use_predict_crp == False:
                    CRP_Content = items[i]['FL_content']
                else:
                    CRP_Content = items[i]['crp_content'] 

                new_items['input'] = self.get_instruction(problem_content[i], new_items['code1'], new_items['code2'], language, is_test,prompt_pattern,CRP_Content) 
                text = tokenizer(new_items['input'],return_tensors='pt')
                if is_test is False and text['input_ids'].shape[1] > 2048: continue

            elif "CRP" == prompt_pattern or "CRFLP" in prompt_pattern:
                CRP_Content = items[i]['FL_content']
                #if len(CRP_Content) > 512:continue
                new_items['input'] = self.get_instruction(problem_content[i], new_items['code1'], None, language, is_test,prompt_pattern,
                CRPContent = CRP_Content) 
                text = tokenizer(new_items['input'],return_tensors='pt')
                if is_test is False and text['input_ids'].shape[1] > 2048: continue

            elif "trace" in prompt_pattern:
                if "base" in prompt_pattern:
                    anno_status = False
                else:
                    anno_status = items[i]['anno_status'][0]

                if "normal" in prompt_pattern:
                    new_items['code2'] = items[i]['code2']
                elif "diff" in prompt_pattern:
                    new_items['code2'] = items[i]['diff_content']
                
                anno_input = items[i]['input']
                anno_output=items[i]['expected_output']
                actual_output = items[i]['actual_output']
                trace=items[i]['anno_code'][0]

                if "CRP" in prompt_pattern or "CRFLP" in prompt_pattern:
                    CRP_Content = items[i]['FL_content']
                    new_items['input'] = self.get_instruction(problem_content[i], new_items['code1'], None, language, is_test,prompt_pattern,CRP_Content,anno_input,anno_output,trace, anno_status, actual_output) 
                    text = tokenizer(new_items['input'],return_tensors='pt')
                    if is_test is False and text['input_ids'].shape[1] > 2048: continue
                    if is_test is True and text['input_ids'].shape[1] > 2048: continue
                else:    
                    new_items['input'] = self.get_instruction(problem_content[i], new_items['code1'], new_items['code2'], 
                    language, is_test,prompt_pattern,None,anno_input,anno_output,trace, anno_status, actual_output) 
                    text = tokenizer(new_items['input'],return_tensors='pt')
                    #normal
                    if is_test == True and text['input_ids'].shape[1] > 2048:
                        continue
                    if is_test == False and text['input_ids'].shape[1] > 2048: #不要了
                        continue
                        
       
            data_list.append(new_items)
            
            
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
                up_data_size = 6#data_size
            else :
                up_data_size = 6
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

        if prompt_pattern == "fixbycrp" or prompt_pattern == "fixbycrflp":
            if use_predict_crp == True:
                crpdata_list =  self.load_json_data(CRPdata_path)
                crp_map = dict()
                for item in crpdata_list:
                    crp_map[item['submission1_id']] = item['crp_content'] #是 crp_content 
                for item in data_list:
                    item['crp_content'] = crp_map[item['submission1_id']]



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
        train_set = TextDataset(all_train_data, tokenizer, is_test, newloss_pattern = args.newloss_pattern)
        return train_set

    
      
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
    
    
    