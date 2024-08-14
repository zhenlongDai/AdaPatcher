import sys

def generate_gdb_script(source_file, output_script='trace.gdb', template_script='trace_template.gdb'):
    with open(source_file, 'r') as f:
        lines = f.readlines()

    with open(template_script, 'r') as f:
        template = f.read()

    with open(output_script, 'w') as f:
        f.write(template)

        # Insert breakpoints for each line
        for i, line in enumerate(lines, 1):
            if line.strip():  # Ignore empty lines
                f.write(f"break {source_file}:{i}\n")
                f.write("commands\n")
                f.write("  silent\n")
                f.write(f"  printf \"At {source_file}:{i}\\n\"\n")
                f.write("  info locals\n")
                f.write("  continue\n")
                f.write("end\n\n")

        f.write("run\n")
        f.write("quit\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_gdb_script.py <source_file>")
        sys.exit(1)

    source_file = sys.argv[1]
    generate_gdb_script(source_file)
    print(f"Generated gdb script for {source_file}")
