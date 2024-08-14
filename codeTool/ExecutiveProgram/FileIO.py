import os
import re
import multiprocessing
from multiprocessing import Manager, Lock
# Manager().dict() 提供了一个共享的字典，可以在多个进程间安全地存储和访问单例实例。Lock 确保只有一个进程能够同时修改共享字典，从而避免竞态条件。
# 在多进程环境中使用 multiprocessing.Manager 和 Lock 实现单例模式时，_instances 和 _lock 是需要在主进程中初始化的。
# 原因是 multiprocessing.Manager 创建的共享对象（如 Manager().dict()) 和 multiprocessing.Lock 需要在主进程中创建，以便它们能够在子进程之间共享。
# 如果不在主进程中初始化这些对象，可能会导致子进程无法正确共享它们。
# 但你可以在类定义中提供初始化方法来初始化这些共享对象，然后在主程序中调用该方法来进行初始化。这样可以使代码更清晰，并确保 _instances 和 _lock 在主进程中正确初始化。
class FileHandlerSingleton:
    # 用于存储每个文件名对应的单例实例 filename对应一个class实体 （P0010)
    _instances = None  # 用于存储每个文件名对应的单例实例
    _lock = None  # 锁对象，用于进程间同步

    @classmethod
    def initialize(cls):
        if cls._instances is None or cls._lock is None:
            manager = Manager()
            cls._instances = manager.dict()
            cls._lock = Lock()

    def __new__(cls, directory):
        abs_directory = os.path.abspath(directory)  # 获取目录的绝对路径
        with cls._lock:
            if abs_directory not in cls._instances:
                instance = super(FileHandlerSingleton, cls).__new__(cls)
                instance.directory = abs_directory
                instance.input_files, instance.output_files = cls.read_text_files(abs_directory)
                # 存储简单的数据结构而不是整个实例
                cls._instances[abs_directory] = {
                    'directory': abs_directory,
                    'input_files': instance.input_files,
                    'output_files': instance.output_files
                }
        return cls._instances[abs_directory]

    def __init__(self, directory):
        with self._lock:
            if not hasattr(self, 'initialized'):
                self.directory = os.path.abspath(directory)  # 获取目录的绝对路径
                self.input_files, self.output_files = self.read_text_files(self.directory)
                self.initialized = True

    @staticmethod
    def read_text_files(directory):
        input_files = {}
        output_files = {}
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory {directory} does not exist")

        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            match = re.match(r'(input|output)\.(\d+)\.txt', filename)
            if match:
                file_type, index = match.groups()
                with open(file_path, 'r') as file:
                    if file_type == 'input':
                        input_files[int(index)] = file.read()
                    elif file_type == 'output':
                        output_files[int(index)] = file.read()

        # 按键排序并转换为有序字典
        input_files = dict(sorted(input_files.items()))
        output_files = dict(sorted(output_files.items()))

        return input_files, output_files

    def get_input_files(self):
        return self.input_files

    def get_output_files(self):
        return self.output_files

def process_function(directory):
    instance_info = FileHandlerSingleton(directory)
    print("Input Files:")
    print(instance_info['input_files'])
    print("Output Files:")
    print(instance_info['output_files'])

if __name__ == '__main__':
    # 初始化共享对象
    FileHandlerSingleton.initialize()

    # 确认工作目录和路径
    test_directory = '/home/develop/dzl/CodeFixProject/CodeDatasets/merged_test_cases/p03391'
    if not os.path.exists(test_directory):
        raise FileNotFoundError(f"Directory {test_directory} does not exist")

    # 创建多个进程
    processes = []
    for i in range(2):  # 使用 2 个进程进行测试
        p = multiprocessing.Process(target=process_function, args=(test_directory,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

