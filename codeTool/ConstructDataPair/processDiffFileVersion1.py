import os
import json
import subprocess
import re

# 检查指定路径是否是一个git仓库
def check_git_repo(path):
    try:
        result = subprocess.run(['git', 'rev-parse', '--is-inside-work-tree'], cwd=path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.returncode == 0 and result.stdout.strip() == 'true'
    except Exception as e:
        return False

def read_json(json_name):
    # 读取用户代码的JSON 数据
    with open(json_name, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 用来处理一个用户的两次提交代码，保存到临时python文件
def save_temp_file(entry, output='output'):
    user_id = entry['user_id']
    problem_id = entry['problem_id']
    
    code1 = entry['code1']
    code2 = entry['code2']

    os.makedirs(f"{output}/{problem_id}", exist_ok=True)
    
    # 创建文件名
    code1_filename = os.path.join(f"{output}/{problem_id}", f'{user_id}_{problem_id}_code1.py')
    code2_filename = os.path.join(f"{output}/{problem_id}", f'{user_id}_{problem_id}_code2.py')
    
    # 将 code1 写入文件
    with open(code1_filename, 'w') as code1_file:
        code1_file.write(code1)
        # 为了解决diff时候的“\ No newline at end of file”问题
        code1_file.write('\n')
    
    # 将 code2 写入文件
    with open(code2_filename, 'w') as code2_file:
        code2_file.write(code2)
        # 为了解决diff时候的“\ No newline at end of file”问题
        code2_file.write('\n')
    
    print(f'************Saved {user_id} code1 to {code1_filename}************')
    print(f'************Saved {user_id} code2 to {code2_filename}************')

    return code1_filename, code2_filename

# 用来对比两个文件
def git_diff_file(code1_filename, code2_filename, output='output_txt', output_indicator_new='+', output_indicator_old='-'):
    # 确保output文件夹存在
    os.makedirs(output, exist_ok=True)
    
    # 确保路径中所有目录名是相对的或完整的路径
    code1_filename = os.path.abspath(code1_filename)
    code2_filename = os.path.abspath(code2_filename)
    output = os.path.abspath(output)
    
    # 将两个文件添加到git仓库中
    subprocess.run(['git', 'add', code1_filename], cwd=output)
    subprocess.run(['git', 'add', code2_filename], cwd=output)
    subprocess.run(['git', 'commit', '-m', f'Add {code1_filename} and {code2_filename}'], cwd=output, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 清空暂存区
    # subprocess.run(['git', 'restore', '--staged', '.'], cwd=output)
    
    # 创建问题ID文件夹
    user_id = code1_filename.split('/')[-1].split('_')[0]
    problem_id = code1_filename.split('/')[-1].split('_')[1]
    problem_output_dir = os.path.join(output, problem_id)
    os.makedirs(problem_output_dir, exist_ok=True)
    
    # 输出文件路径
    output_filename = os.path.join(problem_output_dir, f'{user_id}_{problem_id}.txt')
    
    # 构造git diff命令
    git_diff_command = [
        'git', 'diff', '--no-index', code1_filename, code2_filename,
        f'--output-indicator-new={output_indicator_new}',
        f'--output-indicator-old={output_indicator_old}'
    ]
    
    # 执行git diff命令并通过sed处理输出
    sed_command = "sed 's/^\(@@.*@@\) /\\1\\n/'"
    
    # 执行命令链并将结果写入文件
    with open(output_filename, 'w') as output_file:
        process1 = subprocess.Popen(git_diff_command, stdout=subprocess.PIPE, cwd=output)
        process2 = subprocess.Popen(sed_command, stdin=process1.stdout, stdout=output_file, shell=True)
        process1.stdout.close()  # 允许process1接收SIGPIPE信号
        process2.communicate()
    
    return output_filename    

def process_diff_file(input_file, output_file, new_indicator="+", old_indicator="-"):
    # 打开输入文件和输出文件
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        added_line_written = False

        # 跳过前四行
        """
        前四行是git diff的头部信息，不需要处理。
        """
        for _ in range(4):
            next(infile)

        for line in infile:
            #  如果行以 "+++" 开头，则直接将其写入输出文件
            if line.startswith("+++"):
                outfile.write(line)
                continue
            
            # 如果行以@@开头，则跳过
            if line.startswith("@@"):
                continue
            
            # 处理新增的行
            if line.startswith(new_indicator):
                if not added_line_written:
                    # outfile.write(new_indicator + '\n')
                    outfile.write('<+>' + '\n')
                    added_line_written = True

            # 处理删除的行
            elif line.startswith(old_indicator):
                # outfile.write(line)
                outfile.write('<->' + line[1:])
            # 其余的行保持不变
            else:
                outfile.write(line)
                added_line_written = False

# 处理文件
def dispose_file(data, output='output', output_indicator_new='+', output_indicator_old='-'):
    # 获取输出目录的绝对路径
    output = os.path.abspath(output)
    os.makedirs(output, exist_ok=True)

    # 检查输出目录是否为git仓库
    if not check_git_repo(output):
        subprocess.run(['git', 'init'], cwd=output)
        current_file_path = __file__
        current_file_name = os.path.basename(current_file_path)
        destination_path = os.path.join(output, current_file_name)
        with open(current_file_path, 'r') as source_file:
            with open(destination_path, 'w') as dest_file:
                dest_file.write(source_file.read())
        subprocess.run(['git', 'add', current_file_name], cwd=output)
        subprocess.run(['git', 'commit', '-m', 'Initialize repository'], cwd=output)

    # 处理每个文件, 比较code1和code2
    for i, entry in enumerate(data):
        # 先转化为python文件
        code1_filename, code2_filename = save_temp_file(entry=entry, output=output)
        problem_id = entry["problem_id"]
        # 获取diff后的文件
        output_filename = git_diff_file(code1_filename=code1_filename, code2_filename=code2_filename, output=f"{output}/output_txt", output_indicator_new=output_indicator_new, output_indicator_old=output_indicator_old)
        # 处理diff后的文件
        final_output_filename = os.path.join(output, problem_id, f'{output_filename.split("/")[-1].split(".")[0]}_processed.txt')
        process_diff_file(input_file=output_filename, output_file=final_output_filename, new_indicator=output_indicator_new, old_indicator=output_indicator_old)
        subprocess.run(['git', 'rm', code1_filename], cwd=output)
        subprocess.run(['git', 'rm', code2_filename], cwd=output)
        subprocess.run(['git', 'commit', '-m', f'Remove {code1_filename} and {code2_filename}'], cwd=output)


# 获取增加和删除的行数
def add_empty_line_to_file(filename):
    with open(filename, 'a') as file:
        file.write('\n')

def remove_last_empty_line(file_path):
    """
    打开文件并删除最后一行空行（如果存在）
    :param file_path: 文件路径
    """
    try:
        with open(file_path, 'r+', encoding='utf-8') as file:
            lines = file.readlines()
            if not lines:
                print("The file is empty.")
                return
            
            # 检查最后一行是否为空行
            if lines[-1].strip() == '':
                lines = lines[:-1]
            
            file.seek(0)
            file.truncate()
            file.writelines(lines)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except IOError as e:
        print(f"Error reading/writing file '{file_path}': {e}")
       
def get_diff_stats(code1_filename, code2_filename):
    code1_filename = os.path.abspath(code1_filename)
    code2_filename = os.path.abspath(code2_filename)

    add_empty_line_to_file(code1_filename)
    add_empty_line_to_file(code2_filename)

    result = subprocess.run(
        ['git', 'diff', '--no-index', '--shortstat', code1_filename, code2_filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    output = result.stdout.strip()
    
    # 使用正则表达式解析增加和删除的行数
    insertions = re.search(r'(\d+) insertions?\(\+\)', output)
    deletions = re.search(r'(\d+) deletions?\(-\)', output)
    
    # 如果匹配成功，提取数字，否则设置为0
    added_lines = int(insertions.group(1)) if insertions else 0
    removed_lines = int(deletions.group(1)) if deletions else 0
    
    remove_last_empty_line(code1_filename)
    remove_last_empty_line(code2_filename)

    return added_lines, removed_lines

def get_file_line_count(file_path):
    """
    统计文件的行数
    :param file_path: 文件路径
    :return: 文件行数，如果文件不存在或读取失败，返回0
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            return len(lines)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return 0
    except IOError as e:
        print(f"Error reading file '{file_path}': {e}")
        return 0

if __name__ == '__main__':
    # 读取json文件
    json_name = '/home/develop/dzl/CodeFixProject/CodeFixDatasets/CodeFixPairData/PythonResultData_First/p00004.json'
    data = read_json(json_name)
    # 处理文件
    dispose_file(data, output='output', output_indicator_old='0', output_indicator_new='1')
    # 测试一下get_diff_stats()检测添加了多少行，删除了多少行
    added_lines,removed_lines= get_diff_stats('test1.py', 'test2.py')
    print(f'Added lines: {added_lines}')
    print(f'Removed lines: {removed_lines}')
    # 测试一下get_file_line_count()检测文件有多少行
    line = get_file_line_count('test1.py')
    print(line)
