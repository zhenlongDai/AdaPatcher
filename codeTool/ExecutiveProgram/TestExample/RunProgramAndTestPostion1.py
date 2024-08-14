import sys
import os

# 获取当前文件所在目录的父目录，并将其添加到sys.path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
# sys.path.insert(0, parent_dir)

from codeTool.ExecutiveProgram.FileIO import FileHandlerSingleton
from codeTool.ExecutiveProgram.Worker import Worker, Program_Submission, Checker, Quesion_Test_Point_objectList,Quesion_Test_Point_object


if __name__ == '__main__':
    EXECUTION_ONE = False
    EXECUTION_ALL = True
    CHECK_TEST1 = False
    
    FileDictory = "/home/develop/dzl/CodeFixProject/CodeDatasets/merged_test_cases/p00010"
    
    checker = Checker()
    if CHECK_TEST1 == True:
        str1 = "  Hello, world! \nThis is a test.   \n  "
        str2 = "Hello, world!\nThis is a test.\n"
        result = checker.Check_consistency(str1, str2)
        print(result)  # 输出: True

    w = Worker()
    Compile_File_name = "sxxx.py"
    CodeContent = "\ndef max_sequence_length(X, Y):\n    length = 0\n    current_value = X\n    \n    while current_value <= Y:\n        length += 1\n        current_value *= 2\n        if current_value > Y:\n            break\n        \n        # Now check for the next viable multiple within range\n        next_mult = current_value\n        while next_mult <= Y:\n            current_value = next_mult\n            next_mult *= 2\n        \n    return length\n\n# Input\nX, Y = map(int, input().split())\n\n# Find and print the maximum length of the sequence\nprint(max_sequence_length(X, Y))\n"
    Psubmit = Program_Submission(Compile_File_name, CodeContent)
    if EXECUTION_ONE == True:    
        Test_Point_Input = "1\n0.0 0.0 2.0 0.5127281311709682 2.0 2.0"
        w.Run_Program_By_One_Test_Point(Psubmit, Test_Point_Input)
        print(Psubmit)
        print(type(Psubmit))
    if EXECUTION_ALL == True:
        # 初始化共享对象
        FileHandlerSingleton.initialize()

        # 确认工作目录和路径
        #/data/develop/dzl/CodeFixProject/CodeDatasets/merged_test_cases/p03479/
        test_directory = '/home/develop/dzl/CodeFixProject/CodeDatasets/merged_test_cases/p03479'
        if not os.path.exists(test_directory):
            raise FileNotFoundError(f"Directory {test_directory} does not exist")
        
        Test_List = Quesion_Test_Point_objectList()
        Test_List.inint_Tlist_by_FileHandlerSingleton(FileDirectory=test_directory, deBug= False)
        
        # Q1 = Quesion_Test_Point_object("1\n0.0 0.0 2.0 0.5127281311709682 2.0 2.0", "0.744 1.256 1.460\n\n")
        # Q2 = Quesion_Test_Point_object("1\n0.0 0.0 2.4487612247110815 0.5127281311709682 2.0 2.0", "1.087 0.913 1.420\n\n")
        # Q3 = Quesion_Test_Point_object("1\n0.0 0.0 2.4487612247110815 0.5127281311709682 2.0 2.033916337250816", "1.082 0.936 1.431\n\n")
        # Test_List.Insert_Test_Point(Q1)
        # Test_List.Insert_Test_Point(Q2)
        # Test_List.Insert_Test_Point(Q3)
        
        w.Run_Program_By_One_All_Point(Psubmit, Test_List)
        checker.Check_Run_Result(Psubmit, Test_List)
        print(Psubmit)




