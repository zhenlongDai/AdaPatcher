

export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
CUDA_VISIBLE_DEVICES=0,1,4,5,6,7 torchrun --master_port=63700 --nproc_per_node 6 -m LoraTrainer.eval_Multi \
    --do_eval True \
    --debug_mode True\
    --use_peft True \
    --eval_pattern "train" \
    --prompt_pattern "trace_CRFLP" \
    --data_path "./repairDataset/RepairData-PythonLevel/CRFLPDataset/" \
    --output_dir "./output_dir/loraWeight/trace_CRFLP/checkpoint-14000/" \
    --predict_filePath  "./predict_dir/loraWeight/trace_CRFLP_PerferData/checkpoint-14000_train-prefer-new.json" \
    --per_device_eval_batch_size 1 \
    --do_sample True

