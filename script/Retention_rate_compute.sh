python -m Eval.Retention_rate_compute \
    --code1 "./repairDataset/RepairData-PythonLevel/CRFLPDataset/test.json"\
    --code2 "./predict_evalResult_dir/baseline/twostage_gpt4o.json"\
    --output_file "./predict_evalResult_dir/baseline/retention_rate_dir/twostage_gpt4o.json"\
    --compare_file "./comparebaseline"\
