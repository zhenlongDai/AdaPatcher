#!/bin/bash

# 从命令行参数获取循环次数
CHECKPOINT=14000
PORT=61500
DEVPORT=$((PORT + 10))
gpu_seq2eval=0,3,6,7
PATTERN="train" 
SAVE_STEPS=2000
SAVE_FILE="FLSecondData"
PROMPT_PATTERN="trace_CRFLP"
DATA_FILE="CRFLPDataset"
resume=False
eval_batch_size=2
debug_mode=True

# 获取 GPU 个数
IFS=',' read -ra GPU_ARRAY <<< "$gpu_seq2eval"
gpu_count=${#GPU_ARRAY[@]}

echo "$gpu_count"

# 构建文件路径
file_path="./predict_dir/loraWeight/$SAVE_FILE"
# 检查文件是否存在
if [ -e "$file_path" ]; then
    echo "文件 $file_path 存在"
else
    echo "目录 $file_path 不存在，创建目录"
    mkdir -p "$file_path"
fi

echo ">>>start Eval train to gengeration"
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export NCCL_BLOCKING_WAIT=1
CUDA_VISIBLE_DEVICES=$gpu_seq2eval torchrun --master_port=$PORT --nproc_per_node=$gpu_count -m LoraTrainer.eval_Multi \
      --do_eval True \
      --debug_mode $debug_mode \
      --use_peft True \
      --prompt_pattern "$PROMPT_PATTERN" \
      --eval_pattern "$PATTERN" \
      --output_dir "./output_dir/loraWeight/$PROMPT_PATTERN/checkpoint-$CHECKPOINT/" \
      --data_path "./repairDataset/RepairData-PythonLevel/$DATA_FILE/" \
      --predict_filePath "./predict_dir/loraWeight/$SAVE_FILE/$PATTERN-checkpoint-$CHECKPOINT.json" \
      --per_device_eval_batch_size $eval_batch_size\
      --half True