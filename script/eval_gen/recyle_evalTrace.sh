#!/bin/bash

# 检查是否提供了 upcont 参数
if [ -z "$1" ]; then
  echo "Usage: $0 <upcont>"
  exit 1
fi

UPCONT=$1
PATTERN="dev" 
CHECKPOINT=24000
SAVE_STEPS=2000
FILE_NAME="trace_normal"
PROMPT_PATTERN="trace_normal"

# 设置环境变量
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export NCCL_BLOCKING_WAIT=1
# 循环运行LoraTrainer.eval_Multi
while [ $CHECKPOINT -le $UPCONT ]
do
  echo "Running eval for checkpoint $CHECKPOINT"
  CUDA_VISIBLE_DEVICES=1,2,3 torchrun --master_port=61700 --nproc_per_node=3 -m LoraTrainer.eval_Multi \
      --do_eval True \
      --debug_mode False \
      --use_peft True \
      --prompt_pattern "$PROMPT_PATTERN" \
      --eval_pattern "$PATTERN" \
      --output_dir "./output_dir/loraWeight/$FILE_NAME/checkpoint-$CHECKPOINT/" \
      --data_path "./repairDataset/RepairData-PythonLevel/traceDataset/" \
      --predict_filePath "./predict_dir/loraWeight/$FILE_NAME/$PATTERN-checkpoint-$CHECKPOINT.json" \
      --per_device_eval_batch_size 3 
  
  # 检查上一个命令是否成功
  if [ $? -ne 0 ]; then
    echo "Error running eval for checkpoint $CHECKPOINT. Exiting."
    exit 1
  fi

  # 增加CHECKPOINT的值
  CHECKPOINT=$((CHECKPOINT + SAVE_STEPS))
done

echo "All evaluations completed."
