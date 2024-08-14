#!/bin/bash

# 检查是否提供了 upcont 参数
if [ -z "$1" ]; then
  echo "Usage: $0 <upcont>"
  exit 1
fi

UPCONT=$1
PATTERN=$2 
CHECKPOINT=$3
SAVE_STEPS=$4
SAVE_FILE=$5
# 设置环境变量
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# 循环运行LoraTrainer.eval_Multi
while [ $CHECKPOINT -le $UPCONT ]
do
  echo "Running Execution eval for checkpoint $CHECKPOINT"
  python -m Eval.Eval_Code_Generation-Mprocess \
    --EvalOject_Path "./predict_dir/DpoWeight/$SAVE_FILE/$PATTERN-checkpoint-$CHECKPOINT.json"\
    --Write_prefix_url "./predict_evalResult_dir/$SAVE_FILE/"\
    --test_directory_prefix_url "../CodeDatasets/merged_test_cases/"


  # 检查上一个命令是否成功
  if [ $? -ne 0 ]; then
    echo "Error running eval for checkpoint $CHECKPOINT. Exiting."
    exit 1
  fi

  # 增加CHECKPOINT的值
  CHECKPOINT=$((CHECKPOINT + SAVE_STEPS))
done

echo "All evaluations completed."
