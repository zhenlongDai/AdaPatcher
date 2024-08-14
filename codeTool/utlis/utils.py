import os
import json
import hashlib

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

def calculate_md5(input_string):
    """
    计算并返回字符串的MD5哈希值

    参数:
    input_string (str): 需要计算MD5哈希值的字符串

    返回:
    str: 输入字符串的MD5哈希值
    """
    # 创建一个md5哈希对象
    md5_hash = hashlib.md5()
    
    # 更新哈希对象并计算哈希值
    md5_hash.update(input_string.encode('utf-8'))
    
    # 返回十六进制哈希值
    return md5_hash.hexdigest()

def check_catalogue_exists(filepath):
    """
    检查指定路径的文件是否存在。

    参数:
    filepath (str): 要检查的文件路径。

    返回:
    bool: 如果文件存在，返回True；否则，返回False。
    """
    return  os.path.exists(filepath)


def check_file_exists(filepath):
    """
    检查指定路径的文件是否存在。

    参数:
    filepath (str): 要检查的文件路径。

    返回:
    bool: 如果文件存在，返回True；否则，返回False。
    """
    return  os.path.isfile(filepath)

def read_python_file(file_path):
    """读取指定路径下的Python文件并返回其内容"""
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def write_file_content_to_json(content, json_path):
    """将内容写入指定路径下的JSON文件"""
    data = {'file_content': content}
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def load_list_from_json(input_file_path):
        """从 JSON 文件读取列表"""
        with open(input_file_path, 'r') as json_file:
            data_list = json.load(json_file)
        return data_list
    
def save_list_to_json(lst, filepath):
    """
    将列表存储到指定路径的JSON文件中。
    
    参数:
    lst (list): 需要存储的列表。
    filepath (str): JSON文件的保存路径。
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as json_file:
            json.dump(lst, json_file, ensure_ascii=False, indent=4)
        #print(f"列表已成功保存到 {filepath}")
    except Exception as e:
        print(f"保存列表时出错: {e}")


class CustomJSONEncoder(json.JSONEncoder):
    def encode(self, obj):
        return self.encode_dict(obj)
        

    def encode_dict(self, obj):   
        items = []
        for key, value in obj.items():
            item = f'        "{key}": {json.dumps(value, ensure_ascii=False)}'
            items.append(item)
        return '    {\n' + ',\n'.join(items) + '\n    }'

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory {directory} was created.")
    else:
        print(f"Directory {directory} already exists.")

def save_data_to_json(data, filepath):
    """
    将数据存储到指定路径的JSON文件中，确保特定列表字段不换行，其它字段换行。
    
    参数:
    data (list): 需要存储的列表数据。
    filepath (str): JSON文件的保存路径。
    """
    try:
        ensure_dir(filepath)
        with open(filepath, 'w', encoding='utf-8') as json_file:
            json_file.write('[\n')
            for i, element in enumerate(data):
                json_file.write(CustomJSONEncoder().encode(element))
                if i < len(data) - 1:
                    json_file.write(',\n')
            json_file.write('\n]')
        print(f"数据已成功保存到 {filepath}")
    except Exception as e:
        print(f"保存数据时出错: {e}")

def File2String(file_path, json_output_path):
    
    # 确保输出目录存在，如果不存在则创建
    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)

    content = read_python_file(python_file_path)
    print(content)
    write_file_content_to_json(content, json_output_path)
    print("writing")
if __name__ == '__main__':
    print("here")
    # 指定Python文件路径
    python_file_path = '/home/develop/dzl/CodeFixProject/CodeTool/ConstructDataPair/test2.py'
    # 指定输出的JSON文件路径
    json_output_path = '/home/develop/dzl/CodeFixProject/CodeTool/utlis/out_file/CodeString.json'
    
    File2String(python_file_path ,json_output_path)
    
   

    