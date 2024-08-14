import json

class CustomJSONEncoder(json.JSONEncoder):
    def encode(self, obj):
        return self.encode_dict(obj)
        

    def encode_dict(self, obj):
        
        items = []

        for key, value in obj.items():
            # if key == "code1_test_status":
            #     # 序列化时确保 "code1_test_status" 不换行
            #     item = f'    "{key}": {json.dumps(value, ensure_ascii=False)}'
 
            # else:
            item = f'        "{key}": {json.dumps(value, ensure_ascii=False)}'
            items.append(item)
        return '    {\n' + ',\n'.join(items) + '\n    }'

def save_data_to_json(data, filepath):
    """
    将数据存储到指定路径的JSON文件中，确保特定列表字段不换行，其它字段换行。
    
    参数:
    data (list): 需要存储的列表数据。
    filepath (str): JSON文件的保存路径。
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as json_file:
            json_file.write('[\n')
            for i, element in enumerate(data):
                print(type(element))
                json_file.write(CustomJSONEncoder().encode(element))
                if i < len(data) - 1:
                    json_file.write(',\n')
            json_file.write('\n]')
        print(f"数据已成功保存到 {filepath}")
    except Exception as e:
        print(f"保存数据时出错: {e}")

# 示例用法
data = [{
    "submission1_id": "S324109094",
    "submission2_id": "S459822627",
    "status1": "Wrong Answer",
    "status2": "Accepted",
    "code1": "\n\n\n\nli = []\nfor x in range(3):\n    s = input()\n    s = int(s)\n    li.append(s)\n\n",
    "code2": "\n\n\n\nli = []\nfor x in range(10):\n    s = input()\n    s = int(s)\n    li.append(s)\n\n",
    "original_language1": "Python3",
    "original_language2": "Python3",
    "date1": 1410776608,
    "date2": 1410776776,
    "bleu_score": 0.974890996887752,
    "code1_test_status": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
},{
    "submission1_id": "S324109094",
    "submission2_id": "S459822627",
    "status1": "Wrong Answer",
    "status2": "Accepted",
    "code1": "\n\n\n\nli = []\nfor x in range(3):\n    s = input()\n    s = int(s)\n    li.append(s)\n\n",
    "code2": "\n\n\n\nli = []\nfor x in range(10):\n    s = input()\n    s = int(s)\n    li.append(s)\n\n",
    "original_language1": "Python3",
    "original_language2": "Python3",
    "date1": 1410776608,
    "date2": 1410776776,
    "bleu_score": 0.974890996887752,
    "code1_test_status": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
},{
    "submission1_id": "S324109094",
    "submission2_id": "S459822627",
    "status1": "Wrong Answer",
    "status2": "Accepted",
    "code1": "\n\n\n\nli = []\nfor x in range(3):\n    s = input()\n    s = int(s)\n    li.append(s)\n\n",
    "code2": "\n\n\n\nli = []\nfor x in range(10):\n    s = input()\n    s = int(s)\n    li.append(s)\n\n",
    "original_language1": "Python3",
    "original_language2": "Python3",
    "date1": 1410776608,
    "date2": 1410776776,
    "bleu_score": 0.974890996887752,
    "code1_test_status": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}]

def load_list_from_json(input_file_path):
        """从 JSON 文件读取列表"""
        with open(input_file_path, 'r') as json_file:
            data_list = json.load(json_file)
        return data_list

if __name__ == "__main__":
    save_data_to_json(data, './file.json')
    datalist = load_list_from_json('./file.json')
    for item in datalist:
        print(type(item))
        print(item["code1_test_status"])