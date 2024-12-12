python -m Eval.Retention_rate_compute \
    --code1 "./repairDataset/RepairData-PythonLevel/CRFLPDataset/test.json"\
    --code2 "./rebuttal_file/origin_file/Exec_claude-fewshot.json"\
    --output_file "./rebuttal_file/check_file/Exec_claude-fewshot.json"\
    --compare_file "./rebuttal_file/code/claude-fewshot"\
    --add_flag True

