#!/bin/bash

# # 检查是否提供了 upcont 参数
# if [ -z "$1" ]; then
#   echo "Usage: $0 <upcont> <pattern> <checkpoint> <save_steps> <prompt_pattern> <data_file> <gpu_seq> <port> <second_prompt_pattern> <second_checkpoint>"
#   exit 1
# fi

# 从命令行参数获取值



PATTERN="test"
DATA_FILE="CRFLPDataset"
gpu_seq=0,1,2,3
PORT=62200
batch_size=1
output_dir="./output_dir/DpoWeight/DPOP_Fix_ND3V1/checkpoint-1300"
PROMPT_PATTERN="fixbycrflp"
CRPdata_path="./predict_dir/loraWeight/CRFLP/test-checkpoint-16000.json"
predict_filePath="./predict_dir/Ablation/CC-WT.json"
# 获取 GPU 个数
IFS=',' read -ra GPU_ARRAY <<< "$gpu_seq"
gpu_count=${#GPU_ARRAY[@]}

# 设置环境变量
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export NCCL_BLOCKING_WAIT=1

# 循环运行LoraTrainer.eval_Multi

echo "Running test"
CUDA_VISIBLE_DEVICES=$gpu_seq torchrun --master_port=$PORT --nproc_per_node=$gpu_count -m LoraTrainer.eval_Multi \
    --do_eval True \
    --debug_mode False \
    --use_peft True \
    --prompt_pattern "$PROMPT_PATTERN" \
    --eval_pattern "$PATTERN" \
    --output_dir "$output_dir" \
    --data_path "./repairDataset/RepairData-PythonLevel/$DATA_FILE/" \
    --predict_filePath "$predict_filePath" \
    --per_device_eval_batch_size $batch_size \
    --CRPdata_path "$CRPdata_path"\
    --use_predict_crp True
      
echo "All evaluations completed."
