export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port=60700 --nproc_per_node 4 -m baselineEval.eval_Multi \
    --do_eval True \
    --debug_mode False\
    --use_peft True \
    --eval_pattern "test" \
    --predict_filePath "./predict_dir/baseline/traceBase_baseline_result.json" \
    --data_path "./repairDataset/RepairData-PythonLevel/traceDataset/"\
    --per_device_eval_batch_size 3 \
    --prompt_pattern "trace-normal" \
    --output_dir "./CodeLlama-7b-Instruct-hf/" 
