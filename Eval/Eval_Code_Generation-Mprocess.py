#from ExecutiveProgram.ExecRestRequest import APIManager
from codeTool.ExecutiveProgram.FileIO import FileHandlerSingleton 
from codeTool.ExecutiveProgram.Worker import Checker, Worker, Program_Submission, Quesion_Test_Point_objectList
from codeTool.utlis.utils import load_list_from_json, save_list_to_json, save_data_to_json, check_catalogue_exists, check_file_exists
from tqdm import tqdm
import multiprocessing
import argparse
import os

class EvalProcess:
    def __init__(self, 
        EvalOject_Path,\
        test_directory_prefix_url,\
        Write_prefix_url,\
        language = "Python"):
        self.EvalOject_path = EvalOject_Path
        self.EvalOject_List = load_list_from_json(self.EvalOject_path)

        # 提取文件名并去掉扩展名
        file_name_with_ext = os.path.basename(EvalOject_Path)
        file_name = os.path.splitext(file_name_with_ext)[0]
        self.resotreFile_Path = Write_prefix_url + f"Exec_{file_name}.json"
        self.test_directory_prefix_url = test_directory_prefix_url
        self.language = language
        self.worker = Worker()
        self.checker = Checker()
        FileHandlerSingleton.initialize()     # 初始化文件共享对象

 

    def AddPsubmitResult2item(self, Psubmit, item):

        TotalScore = len(Psubmit.CheckRunResultList)
        # 定义结果到分数的映射
        result_mapping = {
            'Accepted': 1,
            'Time Limit Exceeded': -1,
            'Nonzero Exit Status': -2,
            'Memory Limit Exceeded': -3,
            'Output Limit Exceeded': -4,
            'Wrong Answer': 0
        }
        
        # 使用映射进行转换
        CheckRunResultList = [result_mapping[result] for result in Psubmit.CheckRunResultList]
        # 计算Score
        Score = CheckRunResultList.count(1)
        item["code_test_status"] = CheckRunResultList
        item["code_test_score"] = Score
        item["TotalScore"] = TotalScore
 
        
    def getProblem_id(self, str):
        if str[0] != 'p':
            return f"p{str}"
        else:
            return str
        
    def Process_For_Single_EvalObject(self, item):
        problem_id = self.getProblem_id(item['problem_id'])
        
        test_directory_path = self.test_directory_prefix_url + f"{problem_id}/"
        if check_catalogue_exists(test_directory_path) == False:
            print(f"FilePath {test_directory_path} is not exist")
            return 
        Test_List = Quesion_Test_Point_objectList()
        Test_List.inint_Tlist_by_FileHandlerSingleton(FileDirectory=test_directory_path)
        #跑code_content填充结果
        submission1_id = item["submission1_id"]
        Compile_File_name = f"{submission1_id}.py"
        CodeContent =  item["code_content"]
        Psubmit = Program_Submission(Compile_File_name, CodeContent, self.language)
        self.worker.Run_Program_By_One_All_Point(Psubmit, Test_List, deBug = False)
        self.checker.Check_Run_Result(Psubmit, Test_List)
        self.AddPsubmitResult2item(Psubmit,item)
        return item
    
        
    def ProcessAllData(self):
        with multiprocessing.Pool(processes=8) as pool:  # 设置进程池大小为8
            ResultDataList =  list(tqdm(pool.imap(self.Process_For_Single_EvalObject, self.EvalOject_List), total=len(self.EvalOject_List), desc="Processing elements"))

        save_data_to_json(ResultDataList, self.resotreFile_Path)

    def ProcessAllData_Sequential_Execution(self):
        ResultDataList = []
        for EvalOject in tqdm(self.EvalOject_List, desc="Processing elements"):
            item = self.Process_For_Single_EvalObject(EvalOject)
            ResultDataList.append(item)
            
        save_data_to_json(ResultDataList, self.resotreFile_Path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Code Generation Script")
    parser.add_argument('--EvalOject_Path', type=str, default="", required=True, help='Input data for evaluation')
    parser.add_argument('--test_directory_prefix_url', type=str, default="/data/develop/dzl/CodeDatasets/merged_test_cases/", help='test cases directory')
    parser.add_argument('--Write_prefix_url', type=str, default= "./predict_dir/baseline", help='Number of GPUs to use')
    args = parser.parse_args()

    EXECUTION_ALL = True

    if EXECUTION_ALL == True:
        process = EvalProcess(EvalOject_Path = args.EvalOject_Path, \
                              test_directory_prefix_url = args.test_directory_prefix_url,\
                              Write_prefix_url = args.Write_prefix_url  )
        #process.ProcessAllData_Sequential_Execution()
        process.ProcessAllData()