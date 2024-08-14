import pandas as pd
from collections import defaultdict

import json
from tqdm import tqdm
import re
from ..utlis.utils import load_list_from_json, calculate_md5
from .bleu import code_compute_bleu
# import load_list_from_json, calculate_md5
def remove_comments(code):
    """
    移除代码中的注释，包括单行注释和多行注释
    :param code: 包含注释的代码字符串
    :return: 移除注释后的代码字符串
    """
    # 定义用于匹配单行和多行注释的正则表达式
    single_line_comment_pattern = r'//.*?$|#.*?$'
    multi_line_comment_pattern = r'/\*.*?\*/|\'\'\'.*?\'\'\'|""".*?"""'
    
    # 组合正则表达式
    pattern = re.compile(
        single_line_comment_pattern + '|' + multi_line_comment_pattern,
        re.DOTALL | re.MULTILINE
    )

    # 使用正则表达式移除注释
    cleaned_code = re.sub(pattern, '', code)
    
    return cleaned_code

# 定义一个用户数据结构
class UserData:
    def __init__(self, user_id):
        self.user_id = user_id
        self.records = []

    def add_record(self, record):
        self.records.append(record)




#对一题的数据进行处理
class SingleDataProcess:
    def __init__(self, CSVId, Resotre_File_Path, Language = "Python", Filt_Language = "Python3", prefix_url = "/home/develop/dzl/CodeFixProject/CodeDatasets/Project_CodeNet/"):
        # 创建一个默认字典来存储用户数据
        self.user_data_dict = defaultdict(lambda: UserData(None))
        self.prefix_url = prefix_url + "metadata"
        # 读取CSV文件
        if 'p' not in CSVId:
            CSVId = f"p{CSVId}"
        CSV_Name = f"{self.prefix_url}/{CSVId}.csv"
        self.CSVId = CSVId
        self.Language = Language
        self.File_path = prefix_url + f"data/{CSVId}/{self.Language}"
        self.Resotre_File_Path = Resotre_File_Path
        
        # 构造并存储本题的训练数据
        self.filtered_records = []
        
        df = pd.read_csv(CSV_Name)

        #用pandas内置的方法进行过滤
        # 定义需要保留的status值
        status_values = ['Accepted', 'Wrong Answer', 'Time Limit Exceeded']
        self.filtered_df = df[(df['language'] == Language) & ("Python (2" not in df['original_language']) & ("Python" != df['original_language']) & (df['status'].isin(status_values))].copy()
        #排序 date从小到大
        if len( self.filtered_df) != 0:
            self.filtered_df = self.filtered_df.sort_values(by=['date', 'user_id'], ascending=[True, True]).copy()
            #新加一列对数据最终是否可用的初始化
            self.filtered_df.loc[:, 'useful'] = False
        
        # 遍历数据框中的每一行
        for _, row in self.filtered_df.iterrows():
            user_id = row['user_id']
            if self.user_data_dict[user_id].user_id is None:
                self.user_data_dict[user_id] = UserData(user_id)
            self.user_data_dict[user_id].add_record(row)
            
    def read_python_file(self, submission_id):
        """读取指定路径下的Python文件并返回其内容"""
        if self.Language == "Python":
            file_path = self.File_path + f"/{submission_id}.py"
        
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    
    def save_filtered_df(self, path_str = "./test_filtered_file.csv"):
        # 将过滤后的DataFrame保存到新的CSV文件中
        self.filtered_df.to_csv(path_str, index=False)    
    
    def find_Specific_UserData(self, specific_user_id):
        # 示例：访问特定用户的所有记录
        if specific_user_id in self.user_data_dict:
            user_data = self.user_data_dict[specific_user_id]
            print(f"用户ID: {user_data.user_id}")
            for record in user_data.records:
                print(record)
        else:
            print(f"没有找到用户ID为 {specific_user_id} 的数据")

    #对于单个用户的已过滤wa、tle和ac记录，两两计算相似度。最后从filtered_df中二次过滤可用的数据
    def Construct_Single_user_data(self, specific_user_id):
        #获取数据
        user_data = self.user_data_dict[specific_user_id]
        Len = len(user_data.records)
        #record[j] is AC , record[i] is WA or TLE, they are similar() 
        #每个用户每题只有一对piar数据(i,j)
        s = -1
        t = -1
        maxval = 0
        for j in range(Len):
            
            if user_data.records[j]['status'] != "Accepted": 
                continue #遇到第一个AC的停止
            t = j
            for i in range(0, j):
                if user_data.records[i]['original_language'] != user_data.records[j]['original_language']:continue
                #合理i,j Calculate similarity
                code1 = self.read_python_file(user_data.records[i]['submission_id'])
                code2 = self.read_python_file(user_data.records[j]['submission_id'])
                code1 = remove_comments(code1)
                code2 = remove_comments(code2) #移除代码中的注释
                if code1 == code2: continue 
                if len(code1) == 0 or len(code2) == 0:
                    bleu_score = 0
                else:
                    bleu_score = code_compute_bleu(code1, code2)
                if bleu_score > 0.60 and bleu_score > maxval:
                    maxval = bleu_score
                    s = i
            break #遇到第一个AC找到i后停止
        
        if maxval > 0.60:
            user_data.records[s]['useful'] = True
            user_data.records[t]['useful'] = True
            filtered_record = {
                "user_id":user_data.records[s]['user_id'],
                "problem_id":user_data.records[s]['problem_id'],
                "submission1_id":user_data.records[s]['submission_id'],
                "submission2_id":user_data.records[t]['submission_id'],
                "status1":user_data.records[s]['status'],
                "status2":user_data.records[t]['status'],
                "code1":code1,
                "code2":code2,
                "original_language1":user_data.records[s]['original_language'],
                "original_language2":user_data.records[t]['original_language'],
                "date1":user_data.records[s]['date'],
                "date2":user_data.records[t]['date'],
                "bleu_score":bleu_score,
                "code1_test_status":[],
                "code1_test_score":0
            }
            #print(filtered_record)
            self.filtered_records.append(filtered_record)
            #最大相似度的ACcode和当前wrong code相似度达到要求的视为可接受数据
                
    #构造第一次数据的submission_id对的Set
    def Construct_submissionIDPairRecord_Set(self, first_path):
        file_path = first_path + f"{self.CSVId}.json"
        
        tmp_set = set()
        data_list = load_list_from_json(file_path)
        for item in data_list:
            tmpstr = item["submission1_id"] + item["submission1_id"] 
            tmp_set.add(tmpstr)
        return tmp_set

    #1.过滤Pattern2的数据，2.过滤重复的wa数据 3.构造数据对  
    def Construct_Single_user_data_Pattern2(self, specific_user_id, HaveRecordSet):
   
        #获取数据
        user_data = self.user_data_dict[specific_user_id]
        Len = len(user_data.records)
        
        #record[j] is AC , record[i] is WA, they are similar() 
        s = -1
        t = -1
        AvailableRecordMd5Set = set()
        for j in range(Len):
            
            if user_data.records[j]['status'] != "Accepted": 
                continue #遇到第一个AC的停止
            t = j
            for i in range(0, j):
                if user_data.records[i]['original_language'] != user_data.records[j]['original_language']:continue
                #合理i,j Calculate similarity
                code1 = self.read_python_file(user_data.records[i]['submission_id'])
                code2 = self.read_python_file(user_data.records[j]['submission_id'])
                code1 = remove_comments(code1)
                code2 = remove_comments(code2) #移除代码中的注释
                if code1 == code2: continue 
                if len(code1) == 0 or len(code2) == 0:
                    bleu_score = 0
                else:
                    bleu_score = code_compute_bleu(code1, code2)
                if bleu_score > 0.60 and bleu_score:
                    tmp_str =  user_data.records[i]['submission_id'] + user_data.records[j]['submission_id']
                    if tmp_str in HaveRecordSet: continue
                    md5val = calculate_md5(code1)
                    if md5val not in AvailableRecordMd5Set:
                        AvailableRecordMd5Set.add(md5val)
                        user_data.records[i]['useful'] = True
                        user_data.records[t]['useful'] = True
                        filtered_record = {
                            "user_id":user_data.records[i]['user_id'],
                            "problem_id":user_data.records[i]['problem_id'],
                            "submission1_id":user_data.records[i]['submission_id'],
                            "submission2_id":user_data.records[t]['submission_id'],
                            "status1":user_data.records[i]['status'],
                            "status2":user_data.records[t]['status'],
                            "code1":code1,
                            "code2":code2,
                            "original_language1":user_data.records[i]['original_language'],
                            "original_language2":user_data.records[t]['original_language'],
                            "date1":user_data.records[i]['date'],
                            "date2":user_data.records[t]['date'],
                            "bleu_score":bleu_score,
                            "code1_test_status":[],
                            "code1_test_score":0
                        }
                        #print(filtered_record)
                        self.filtered_records.append(filtered_record)
            break #遇到第一个AC找到i后停止


    #2.对于每个用户的构造合理数据，然后保存该题目的可用记录形式
    def Construct_All_user_data(self, pattern = "First", first_path = "/home/develop/dzl/CodeFixProject/CodeFixDatasets/CodeFixPairData/PythonResultData_First/"):
        #遍历处理每个user的数据
        if pattern == "First":
            for k, v in self.user_data_dict.items():
                self.Construct_Single_user_data(k)
        else:
            for k, v in self.user_data_dict.items():
                HaveRecordSet = self.Construct_submissionIDPairRecord_Set(first_path)
                self.Construct_Single_user_data_Pattern2(k, HaveRecordSet)
        # 指定保存的文件名
        Resotre_File_name = self.Resotre_File_Path + f"{self.CSVId}.json"

        # 将列表对象保存为 JSON 文件
        with open(Resotre_File_name, 'w') as json_file:
            json.dump(self.filtered_records, json_file, indent=4)  # indent 参数用于格式化输出，使其更加美观
        
        
    
    
#一次过滤路径 Program_Question_Data/Available_PId.json 有p   
#二次过滤路径 Program_Question_Data/Available_Filt_PId.json 缺p
class AllDataProcess: 
    def __init__(self, Available_PId_path = "/home/develop/dzl/CodeFixProject/CodeFixDatasets/Program_Question_Data/Available_Filt_PId.json",\
        Resotre_File_Path = "/home/develop/dzl/CodeFixProject/CodeFixDatasets/CodeFixPairData/PythonData/"):
        self.Available_PId_Set = self.load_list_from_json(Available_PId_path)
        self.Resotre_File_Path = Resotre_File_Path 
        
    def load_list_from_json(self, input_file_path):
        """从 JSON 文件读取列表"""
        with open(input_file_path, 'r') as json_file:
            data_list = json.load(json_file)
        return data_list
    
    def ProcessAllData(self, pattern = "First"):
        for Pid in tqdm(self.Available_PId_Set, desc="Processing elements"):
            Single_data_process = SingleDataProcess(Pid, self.Resotre_File_Path)
            Single_data_process.Construct_All_user_data(pattern)


if __name__ == "__main__":
    #根据pId,存储路径得到该题目的记录
    #data_process = SingleDataProcess("00025","/home/develop/dzl/CodeFixProject/CodeFixDatasets/CodeFixPairData/")
    #data_process.Construct_All_user_data()
    #data_process.Construct_Single_user_data("u766477342")
    #data_process.save_filtered_df()

    alldataprocess = AllDataProcess()
    alldataprocess.ProcessAllData(pattern = "Second")
    
    
    