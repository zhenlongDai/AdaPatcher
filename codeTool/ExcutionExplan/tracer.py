import sys
F_name = ""
def trace_lines(frame, event, arg):
    if event != "line":
        return trace_lines
    co = frame.f_code
    func_name = co.co_name
    line_no = frame.f_lineno
    filename = co.co_filename
    
    
    
    # 读取当前执行的代码行
    try:
        with open(target_script, "r") as file:
            lines = file.readlines()
            current_line = lines[line_no - 1].strip()
    except Exception as e:
        current_line = f"<unable to read line: {e}>"
    func_name_val = f"{func_name}"
    if func_name_val == "<module>":
        
        # 使用 getattr 来避免初始未定义的错误
        last_locals = getattr(trace_lines, 'last_locals', {})
        current_locals = frame.f_locals.copy()

        # 检查是否有上一行的行号记录，如果没有初始化为 None
        last_line_no = getattr(trace_lines, 'last_line_no', None)
        
        

        # 比较当前行和上一行的局部变量值
        changed_vars = {}
        for var, value in current_locals.items():
            if var not in last_locals or last_locals[var] != value:
                changed_vars[var] = value

        # 打印变化的变量和当前行号
        if changed_vars:
            print(f"Function {func_name} Line {line_no} {current_line}: Variables changed - {changed_vars}")
            # 打印当前行和上一行的行号
            print(f"Function {func_name} Line {line_no}, Last Line {last_line_no}")
    
        
        #print(f"******{func_name} Line {line_no}")
        #print(f"******{func_name} Line {line_no}: {current_line} | Locals: {frame.f_locals}")
        #print(f"******{func_name} Line {line_no}: Locals: {frame.f_locals}")
        #print(f"{current_line}")
    
        # 更新上一行的局部变量状态
        trace_lines.last_locals = current_locals
        # 更新存储的上一行行号为当前行号
        trace_lines.last_line_no = line_no
    return trace_lines

def run_script(filename, input_file):
    print("start")
    with open(filename) as file, open(input_file, 'r') as input_data:
        # 重定向标准输入到文件
        sys.stdin = input_data
        script_content = file.read()
        # 执行脚本
        exec(script_content)
        # 恢复标准输入
        sys.stdin = sys.__stdin__

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python tracer.py <script_to_trace.py> <input_file>")
    else:
        target_script = sys.argv[1]
        input_file = sys.argv[2]
        sys.settrace(trace_lines)
        run_script(target_script, input_file)
        sys.settrace(None)
