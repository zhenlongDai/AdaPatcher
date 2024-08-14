#!/bin/bash

# 从命令行参数获取循环次数
begin_iterations=2000
end_iterations=10000
PORT=62600
DEVPORT=$((PORT + 10))
gpu_seq=1,2,4,5
gpu_seq2eval=1,2,4,5
gpu_seq2test=1,2,4,5
PATTERN="dev" 
SAVE_STEPS=2000

PROMPT_PATTERN="fixbycrflp"
DATA_FILE="CRFLPDataset"
resume=False
num_train_epochs=10
train_batch_size=4
eval_batch_size=2
test_batch_size=1
newloss_pattern=False
debug_mode=False
train_status=false
dev_status=false
Exec_status=false
statistic_status=true
test_status=true

# 获取 GPU 个数
IFS=',' read -ra GPU_ARRAY <<< "$gpu_seq"
gpu_count=${#GPU_ARRAY[@]}

if [ "$train_status" = "true" ]; then
    echo ">>>start Train"
    bash ./script/pipeline/part/train.sh "$gpu_seq" "$PROMPT_PATTERN" "$DATA_FILE" "$PORT" "$gpu_count" "$resume" "$num_train_epochs" "$newloss_pattern" "$debug_mode"\
    "$SAVE_STEPS" "$train_batch_size"
fi



if [ "$dev_status" = "true" ]; then

    # 构建文件路径
    file_path="./predict_dir/loraWeight/$PROMPT_PATTERN"
    # 检查文件是否存在
    if [ -e "$file_path" ]; then
        echo "文件 $file_path 存在"
    else
        echo "目录 $file_path 不存在，创建目录"
        mkdir -p "$file_path"
    fi

    echo ">>>start Eval DEV to gengeration"
    bash ./script/pipeline/part/recyle_eval.sh "$end_iterations" "$PATTERN" "$begin_iterations" "$SAVE_STEPS" "$PROMPT_PATTERN" "$DATA_FILE" "$gpu_seq2eval" "$DEVPORT" "$eval_batch_size"
fi


if [ "$Exec_status" = "true" ]; then
    # 构建文件路径
    file_path="./predict_evalResult_dir/$PROMPT_PATTERN"
    # 检查文件是否存在
    if [ -e "$file_path" ]; then
        echo "文件 $file_path 存在"
    else
        echo "目录 $file_path 不存在，创建目录"
        mkdir -p "$file_path"
    fi

    echo ">>>start Execution" 
    bash ./script/pipeline/part/recyle_Execution.sh "$end_iterations" "$PATTERN" "$begin_iterations" "$SAVE_STEPS" "$PROMPT_PATTERN"
fi

if [ "$statistic_status" = "true" ]; then
    # 初始化最大 rate 和对应的 i
    max_rate=0.0
    best_iteration=0
    echo ">>>start choose best dev rate value"
    # 遍历迭代范围，执行脚本并捕获输出
    for ((i=begin_iterations; i<=end_iterations; i+=SAVE_STEPS)); do
        output=$(bash ./script/pipeline/part/Statistical_Execution_Results.sh "$PROMPT_PATTERN" "$PATTERN" "$i")
        echo "$output"
        #从输出中提取 rate 值
        avg_improve_rate=$(echo "$output" | grep 'avg_improve_rate = ' | awk '{print $3}')
        # 检查 rate 是否提取成功
        if [ -z "$avg_improve_rate" ]; then
            echo "Failed to extract rate for iteration $i"
            continue
        fi
        #echo eavg_improve_rate
        # 比较并更新最大 rate 和对应的 i
        if (( $(echo "$avg_improve_rate > $max_rate" | bc -l) )); then
        max_rate=$avg_improve_rate
        best_iteration=$i
        fi
    done
fi

if [ "$test_status" = "true" ]; then
    echo ">>>start Eval TEST to gengeration  $best_iteration"
    bash ./script/pipeline/part/recyle_eval.sh "$best_iteration" "test" "$best_iteration" "$SAVE_STEPS" "$PROMPT_PATTERN" "$DATA_FILE" "$gpu_seq2test" "$DEVPORT" "$test_batch_size"
    echo ">>>start Execution For Test" 
    bash ./script/pipeline/part/recyle_Execution.sh "$best_iteration" "test" "$best_iteration" "$SAVE_STEPS" "$PROMPT_PATTERN"
    echo ">>>start Statistic Execution Result For test"
    bash ./script/pipeline/part/Statistical_Execution_Results.sh "$PROMPT_PATTERN" "test" "$best_iteration"
fi