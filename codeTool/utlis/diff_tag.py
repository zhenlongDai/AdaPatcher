import json

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_user_ids(data):
    return {item['problem_id'] for item in data}

def find_unique_user_ids(file1_path, file2_path):
    data1 = load_json(file1_path)
    data2 = load_json(file2_path)

    user_ids1 = extract_user_ids(data1)
    user_ids2 = extract_user_ids(data2)

    unique_user_ids = user_ids1.symmetric_difference(user_ids2)
    return unique_user_ids

if __name__ == "__main__":
    file1_path = '/data/develop/dzl/CodeRepairLLM-Speed/repairDataset/RepairData-PythonLevel/dev.json'
    file2_path = '/data/develop/dzl/CodeRepairLLM-Speed/predict_evalResult_dir/6-14/Exec_dev-checkpoint-4000.json'

    unique_user_ids = find_unique_user_ids(file1_path, file2_path)
    print(f'User IDs not present in both files: {unique_user_ids}')
