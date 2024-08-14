#!/bin/bash

# 从命令行参数获取循环次数
begin_iterations=100
end_iterations=1200
PORT=61600
DEVPORT=$((PORT + 10))
gpu_seq=4,5,6,7
gpu_seq2eval=4,5,6,7
gpu_seq2test=4,5,6,7
PATTERN="dev" 
SAVE_STEPS=100
eval_steps=100
Version="DPOP_FixBase_V1"
model_name_or_path="./output_dir/DirectFixCodeLlama"
SECOND_PROMPT_PATTERN="fixbycrflp"
PerferDATA_FILE="data/FixPerferdataset3"
DATA_FILE="mixCRFLPDataset"
test_CRPdata_path="./predict_dir/loraWeight/trace_CRFLP/test-checkpoint-14000.json"
resume=False
num_train_epochs=10
train_batch_size=2
eval_batch_size=2
test_batch_size=1 
gradient_accumulation_steps=2
train_mode="DPOP"

debug_mode=False
train_status=false
dev_second_status=false
Exec_status=false
statistic_status=true
test_status=true

# 获取 GPU 个数
IFS=',' read -ra GPU_ARRAY <<< "$gpu_seq"
gpu_count=${#GPU_ARRAY[@]}

if [ "$train_status" = "true" ]; then
    # 构建文件路径
    file_path="./output_dir/DpoWeight/$Version"
    # 检查文件是否存在
    if [ -e "$file_path" ]; then
        echo "文件 $file_path 存在"
    else
        echo "目录 $file_path 不存在，创建目录"
        mkdir -p "$file_path"
    fi
    #echo ">>>start Train"
    #bash ./script/pipeline/part/train.sh "$gpu_seq" "$PROMPT_PATTERN" "$DATA_FILE" "$PORT" "$gpu_count" "$resume" "$num_train_epochs" "$newloss_pattern" "$debug_mode" "$eval_steps" "$train_batch_size"
    export NCCL_P2P_DISABLE=1
    export CUDA_VISIBLE_DEVICES=$gpu_seq
    ACCELERATE_LOG_LEVEL=info accelerate launch -m DpoTrainer.dpo_llama \
        --model_name_or_path "$model_name_or_path" \
        --output_dir "./output_dir/DpoWeight/$Version"\
        --data_path "./repairDataset/RepairData-PythonLevel/$PerferDATA_FILE/"\
        --save_steps $SAVE_STEPS \
        --num_train_epochs $num_train_epochs \
        --per_device_train_batch_size $train_batch_size \
        --per_device_eval_batch_size $train_batch_size\
        --run_name "Dpo_CodeLLama_$Version" \
        --gradient_accumulation_steps "$gradient_accumulation_steps" \
        --resume "$resume"\
        --train_mode "$train_mode"\
        --prompt_pattern "$SECOND_PROMPT_PATTERN"
fi


if [ "$dev_second_status" = "true" ]; then
    # 构建文件路径
    file_path="./predict_dir/DpoWeight/$Version-GEN"
    # 检查文件是否存在
    if [ -e "$file_path" ]; then
        echo "文件 $file_path 存在"
    else
        echo "目录 $file_path 不存在，创建目录"
        mkdir -p "$file_path"
    fi

    echo ">>>start eval dev to fix buggy code by fixModel"
    bash ./script/pipeline/part/recyle_eval2PerferFix.sh "$end_iterations" "$PATTERN" "$begin_iterations" "$SAVE_STEPS" "$SECOND_PROMPT_PATTERN" \
    "$DATA_FILE" "$gpu_seq2eval" "$DEVPORT" "$eval_batch_size" "$Version" "False" "None"
fi

if [ "$Exec_status" = "true" ]; then
    # 构建文件路径
    file_path="./predict_evalResult_dir/$Version-GEN"
    # 检查文件是否存在
    if [ -e "$file_path" ]; then
        echo "文件 $file_path 存在"
    else
        echo "目录 $file_path 不存在，创建目录"
        mkdir -p "$file_path"
    fi
    echo ">>>start Execution" 
    bash ./script/pipeline/part/recyle_Execution2Perfer.sh "$end_iterations" "$PATTERN" "$begin_iterations" "$SAVE_STEPS" "$Version-GEN"
fi
#需要通过genbycpr执行得到结果

if [ "$statistic_status" = "true" ]; then
    #初始化最大 rate 和对应的 i
    max_rate=0.0
    best_iteration=0
    echo ">>>start choose best dev rate value"
    # 遍历迭代范围，执行脚本并捕获输出
    for ((i=begin_iterations; i<=end_iterations; i+=SAVE_STEPS)); do
        output=$(bash ./script/pipeline/part/Statistical_Execution_Results.sh "$Version-GEN" "$PATTERN" "$i")
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
echo ">>>> $best_iteration"

if [ "$test_status" = "true" ]; then

    echo ">>>start Eval TEST to gengeration  $best_iteration"
    
    bash ./script/pipeline/part/recyle_eval2PerferFix.sh "$best_iteration" "test" "$best_iteration" "$SAVE_STEPS" "$SECOND_PROMPT_PATTERN"\
     "$DATA_FILE" "$gpu_seq2test" "$DEVPORT" "$test_batch_size" "$Version" "True" "$test_CRPdata_path"
   
    echo ">>>start Execution For TESt" 
    bash ./script/pipeline/part/recyle_Execution2Perfer.sh "$best_iteration" "test" "$best_iteration" "$SAVE_STEPS" "$Version-GEN"
    echo ">>>start Statistic Execution Result For TEST"
    bash ./script/pipeline/part/Statistical_Execution_Results.sh "$Version-GEN" "test" "$best_iteration"
fi