import requests
import json
import copy
# 目标 URL 前缀
#url_prefix = 'http://192.168.7.151:5050/'
url_prefix = 'http://127.0.0.1:5050/'
class APIManager: #可重入
    def __init__(self, url_prefix):
        self.url_prefix = url_prefix
        # 用于POST请求的JSON数据
        self.CompileData = {
            "cmd": [{
                "args": ["python3", "-m", "py_compile" ,"pytest.py"],
                "env": ["PATH=/usr/bin:/bin"],
                "files": [{
                    "content": ""
                }, {
                    "name": "stdout",
                    "max": 10240
                }, {
                    "name": "stderr",
                    "max": 10240
                }],
                "cpuLimit": 10000000000,
                "memoryLimit": 1048576000,
                "procLimit": 50,
                "copyIn": {
                    # "pytest.py": {
                    #     "content": "for i in range(int(input())):\n    x1, y1, x2, y2, x3, y3 = map(float, input().split())\n    c = (x1-x2)**2 + (y1-y2)**2\n    a = (x2-x3)**2 + (y2-y3)**2\n    b = (x3-x1)**2 + (y3-y1)**2\n    # 16s^2\n    s = 2*(a*b + b*c + c*a) - (a*a + b*b + c*c)\n\n    px = (a*(b+c-a)*x1 + b*(c+a-b)*x2 + c*(a+b-c)*x3) / s\n    py = (a*(b+c-a)*y1 + b*(c+a-b)*y2 + c*(a+b-c)*y3) / s\n\n    ar = a**0.5\n    br = b**0.5\n    cr = c**0.5\n\n    r = ar*br*cr / ((ar+br+cr)*(-ar+br+cr)*(ar-br+cr)*(ar+br-cr))**0.5\n\n    print(\"{:>.3f}\".format(px),\"{:>.3f}\".format(py),\"{:>.3f}\".format(r))"
                    # }
                },
                "copyOut": ["stdout", "stderr"],
                "copyOutCached": ["__pycache__/pytest.cpython-311.pyc"]
            }]
        }
        self.ExecutiveData = {
            "cmd": [{
                "args": ["/usr/bin/python3", "__pycache__/pytest.cpython-311.pyc"],
                "env": ["PATH=/usr/bin:/bin"],
                "files": [{
                    "content": "1\n0.0 0.0 2.0 0.5127281311709682 2.0 2.0"
                }, {
                    "name": "stdout",
                    "max": 10240
                }, {
                    "name": "stderr",
                    "max": 10240
                }],
                "cpuLimit": 1000000000,
                "memoryLimit": 104857600,
                "procLimit": 50,
                "copyIn": {
                    "__pycache__/pytest.cpython-311.pyc": {
                        "fileId": "BWO4DDB5"
                    }
                }
            }]
        }
        self.CopyIn_FileId_Dict = self.GET_File_Id_Dict()
        
    # 发送GET请求获取文件ID SET
    def GET_File_Id_Dict(self, deBug = False):
        
        request_rul = self.url_prefix + f"file"
        response = requests.get(request_rul)
        data_dict = json.loads(response.text)
        if deBug == True:
            print('>>> GET Request Response:')
            print('>>> Status Code:', response.status_code) #200
            print('>>> Response Body:', response.text)
        return data_dict
    
    # 发送DELETE请求删除指定FileId的文件
    def Delete_File_By_Fileid(self, FileId):
        
        #print('\n>>> DELETE Request Response:')
        request_rul = self.url_prefix + f"file/{FileId}"
        response = requests.delete(request_rul)
        #print('>>> Status Code:', response.status_code) #200
    
    # 发送Post_Json_Text的POST请求
    def send_post_request(self, Post_Json_Data, deBug = False):
        
        request_rul = self.url_prefix + f"run"
        response = requests.post(request_rul, json= Post_Json_Data)
        if deBug == True:
            print('\nPOST Request Response:')
            print('Status Code:', response.status_code)
            print('Response Body:', response.text)
        return response.text
    
    def Modify_Command_For_Compile_Program(self, new_args, code_content, language = "Python"):
        data_copy = copy.deepcopy(self.CompileData)
        
        if language == "Python":
            Execute_File_name = new_args[-1]
            # 修改args
            data_copy["cmd"][0]["args"] = new_args
            # 修改copyIn中文件的内容
            # 没有Execute_File_name键，则创建它
            data_copy["cmd"][0]["copyIn"][Execute_File_name] = {"content": code_content}
            Execute_File_name_without_type = Execute_File_name.split(".")[0]
            #print(Execute_File_name_without_type)
            data_copy["cmd"][0]["copyOutCached"] = [f"__pycache__/{Execute_File_name_without_type}.cpython-311.pyc"]
        else:
            Execute_File_name = "" # None
        return data_copy    
    
    def Modify_Command_For_Execute_Program(self, new_args, CopyIn_fileId, Execute_File_name, TestPoint_Content, language = "Python"):
        data_copy = copy.deepcopy(self.ExecutiveData)
        
        if language == "Python":
            Execute_File_name = new_args[-1]
            # 修改args
            data_copy["cmd"][0]["args"] = new_args
            # 修改copyIn中文件的内容
            # 没有Execute_File_name键，则创建它
            data_copy["cmd"][0]["files"][0]["content"] = TestPoint_Content
            data_copy["cmd"][0]["copyIn"] = {Execute_File_name:{"fileId": CopyIn_fileId}}
        else:
            Execute_File_name = "" # None
        return data_copy
    
    # 通过Fileid删除所有的缓存文件
    def Delete_All_File(self, deBug = False):
        
        #获取Fileid
        Fileid_dict = self.GET_file_id_dict()
        #调用请求删除对应Fileid的文件
        for k, v in Fileid_dict.items():
            if "pyc" not in v:
                self.Delete_file_by_fileid(k)   
        print(">>> Delete Success")
        
    # 通过Fileid删除指定的缓存文件
    def Delete_All_File_by_FileId(self, FileId, deBug = False): 
        self.Delete_file_by_fileid(FileId)   
        print(f">>> Delete FileId:{FileId} Success")
    
    
    # 编译对应language的CodeString) 返回Result Jons(include status/fileIds)
    def Compile_Program(self, Execute_File_name, CodeContent, language = "Python"):
        #构造REST的json数据
        if language == "Python":
            new_args = ["python3", "-m", "py_compile" , f"{Execute_File_name}"]
            Post_Json_Data = self.Modify_Command_For_Compile_Program(new_args, CodeContent)
            Json_Text_response = self.send_post_request(Post_Json_Data)
        else:
            raise ValueError(">>> language value is wrong")
        
        return Json_Text_response
    
    
    #  执行程序（默认已经编译成功）
    def Execute_Program_After_Compile(self, CopyIn_fileId, Execute_File_name, TestPoint_Content, language = "Python") :
        
        if language == "Python":
            #构造REST的json数据
            new_args = ["/usr/bin/python3", Execute_File_name]
            #基于Post_Json_Data发送请求
            Post_Json_Data = self.Modify_Command_For_Execute_Program(new_args, CopyIn_fileId, Execute_File_name, TestPoint_Content)
            #请求执行
            Json_Text_response = self.send_post_request(Post_Json_Data)
            #请求执行完所有测试点后，删除对应缓存文件
            #self.Delete_All_File_by_FileId(CopyIn_fileId)    
        else:
            raise ValueError(">>> language value is wrong")
            
        return Json_Text_response


if __name__ == '__main__':
    TEST_COMPILE = True
    TEST_GET_FILE_ID = False
    TEST_EXCUTE_FILE = False
    TEST_COMPILE_AND_EXCUTE = True
    api_manager = APIManager(url_prefix)
    #api_manager.send_post_request(api_manager.data)
    
    #测试编译功能
    if TEST_COMPILE == True:
        Compile_File_name = "s001.py"
        CodeContent = "for i in range(int(input())):\n    x1, y1, x2, y2, x3, y3 = map(float, input().split())\n    c = (x1-x2)**2 + (y1-y2)**2\n    a = (x2-x3)**2 + (y2-y3)**2\n    b = (x3-x1)**2 + (y3-y1)**2\n    # 16s^2\n    s = 2*(a*b + b*c + c*a) - (a*a + b*b + c*c)\n\n    px = (a*(b+c-a)*x1 + b*(c+a-b)*x2 + c*(a+b-c)*x3) / s\n    py = (a*(b+c-a)*y1 + b*(c+a-b)*y2 + c*(a+b-c)*y3) / s\n\n    ar = a**0.5\n    br = b**0.5\n    cr = c**0.5\n\n    r = ar*br*cr / ((ar+br+cr)*(-ar+br+cr)*(ar-br+cr)*(ar+br-cr))**0.5\n\n    print(\"{:>.3f}\".format(px),\"{:>.3f}\".format(py),\"{:>.3f}\".format(r))"
        api_manager.GET_File_Id_Dict(deBug = True)
        
        result = api_manager.Compile_Program(Compile_File_name, CodeContent)
        print(result)
        api_manager.GET_File_Id_Dict(deBug = True)
    #测试获取已编译文件
    if TEST_GET_FILE_ID == True:
        api_manager.GET_File_Id_Dict(deBug = True)
    #测试执行已编译文件
    if TEST_EXCUTE_FILE == True:
        #  执行程序（默认已经编译）
        #def Execute_Program_After_Complie(self, CopyIn_fileId, Execute_File_name, TestPoint_Content, language = "Python") :
        CopyIn_fileId = "MFM7PJOD"
        Execute_File_name = "__pycache__/s001.cpython-311.pyc"
        TestPoint_Content = "1\n0.0 0.0 2.0 0.5127281311709682 2.0 2.0"
        Json_Text_response = api_manager.Execute_Program_After_Compile(CopyIn_fileId, Execute_File_name, TestPoint_Content)
        print(Json_Text_response)
    #if  TEST_COMPILE_AND_EXCUTE == True:
        
    #send_post_request(url, data)
    #send_delete_request(url)
