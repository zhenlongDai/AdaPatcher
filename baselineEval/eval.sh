export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
CUDA_VISIBLE_DEVICES=7 torchrun --master_port=63300 --nproc_per_node 1 -m baselineEval.eval \
    --do_eval True \
    --debug_mode False\
    --predict_dir "./predict_dir/baseline/trace_baseline_result.json" \
    --data_path "./repairDataset/RepairData-PythonLevel/traceDataset/"\
    --per_device_eval_batch_size 1 \
    --prompt_pattern "trace-normal" 
