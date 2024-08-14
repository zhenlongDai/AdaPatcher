#!/bin/bash

# # 检查是否提供了 upcont 参数
# if [ -z "$1" ]; then
#   echo "Usage: $0 <upcont> <pattern> <checkpoint> <save_steps> <prompt_pattern> <data_file> <gpu_seq> <port> <second_prompt_pattern> <second_checkpoint>"
#   exit 1
# fi

# 从命令行参数获取值



UPCONT=$1
PATTERN=$2 
CHECKPOINT=$3
SAVE_STEPS=$4
PROMPT_PATTERN=$5
DATA_FILE=$6
gpu_seq=$7
PORT=$8
batch_size=$9
SECOND_PROMPT_PATTERN_NAME=${10}
SECOND_CHECKPOINT=${11}
Version=${12}
#SECOND_PROMPT_PATTERN="fixbycrp"
#SECOND_CHECKPOINT=26000

# 检查是否所有参数都已提供
if [ -z "$UPCONT" ] || [ -z "$PATTERN" ] || [ -z "$CHECKPOINT" ] || [ -z "$SAVE_STEPS" ] || [ -z "$PROMPT_PATTERN" ] || [ -z "$DATA_FILE" ] || [ -z "$gpu_seq" ] || [ -z "$PORT" ] || [ -z "$SECOND_PROMPT_PATTERN_NAME" ] || [ -z "$SECOND_CHECKPOINT" ]; then
  echo "Missing one or more required parameters."
  echo "Usage: $0 <upcont> <pattern> <checkpoint> <save_steps> <prompt_pattern> <data_file> <gpu_seq> <port> <second_prompt_pattern_name> <second_checkpoint>"
  exit 1
fi
# 获取 GPU 个数
IFS=',' read -ra GPU_ARRAY <<< "$gpu_seq"
gpu_count=${#GPU_ARRAY[@]}

# 设置环境变量
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export NCCL_BLOCKING_WAIT=1

# 循环运行LoraTrainer.eval_Multi
while [ $CHECKPOINT -le $UPCONT ]
do
  echo "Running eval for checkpoint $CHECKPOINT"
  CUDA_VISIBLE_DEVICES=$gpu_seq torchrun --master_port=$PORT --nproc_per_node=$gpu_count -m LoraTrainer.eval_Multi \
      --do_eval True \
      --debug_mode False \
      --use_peft True \
      --prompt_pattern "$PROMPT_PATTERN" \
      --eval_pattern "$PATTERN" \
      --output_dir "./output_dir/loraWeight/$SECOND_PROMPT_PATTERN_NAME/checkpoint-$SECOND_CHECKPOINT/" \
      --data_path "./repairDataset/RepairData-PythonLevel/$DATA_FILE/" \
      --predict_filePath "./predict_dir/DpoWeight/$Version-GEN/$PATTERN-checkpoint-$CHECKPOINT.json" \
      --per_device_eval_batch_size $batch_size \
      --CRPdata_path "./predict_dir/DpoWeight/$Version/$PATTERN-checkpoint-$CHECKPOINT.json"\
      --use_predict_crp True
      
  # 检查上一个命令是否成功
  if [ $? -ne 0 ]; then
    echo "Error running eval for checkpoint $CHECKPOINT. Exiting."
    exit 1
  fi

  # 增加CHECKPOINT的值
  CHECKPOINT=$((CHECKPOINT + SAVE_STEPS))
done

echo "All evaluations completed."
