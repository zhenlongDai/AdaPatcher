import sys

def trace_lines(frame, event, arg):
    if event != "line":
        return trace_lines
    co = frame.f_code
    func_name = co.co_name
    line_no = frame.f_lineno
    filename = co.co_filename

    # 保证只对目标脚本进行追踪
    if "target_script.py" in filename:
        # 检查是否已经缓存了文件的内容
        if not hasattr(trace_lines, "file_lines") or trace_lines.current_file != filename:
            try:
                with open(filename, "r") as file:
                    trace_lines.file_lines = file.readlines()
                    trace_lines.current_file = filename
            except Exception as e:
                trace_lines.file_lines = []
                print(f"Error reading file {filename}: {e}")

        # 获取并打印当前执行的代码行
        current_line = trace_lines.file_lines[line_no - 1].strip() if line_no <= len(trace_lines.file_lines) else "<line not found>"
        print(f"{func_name} Line {line_no}: {current_line} | Locals: {frame.f_locals}")

    return trace_lines

def run_script(filename, input_file):
    with open(filename) as file, open(input_file, 'r') as input_data:
        sys.stdin = input_data
        script_content = file.read()
        exec(script_content)
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
