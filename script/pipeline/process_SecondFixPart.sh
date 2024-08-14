#!/bin/bash
# 混合数据训练第二阶段【None,微调，二阶段最好模型】
# 从命令行参数获取循环次数
begin_iterations=2000
end_iterations=30000
PORT=62100
DEVPORT=$((PORT + 10))
gpu_seq=0,1,2,4
gpu_seq2eval=0,1,2,4
gpu_seq2test=0,1,2,4
PATTERN="dev" 
SAVE_STEPS=2000


#dev_CRPdata_path="./predict_dir/loraWeight/trace_CRFLP/dev-checkpoint-14000.json"
test_CRPdata_path="./predict_dir/loraWeight/trace_CRFLP/test-checkpoint-14000.json"
#test_CRPdata_path="./predict_dir/loraWeight/CRFLP/test-checkpoint-16000.json"
INPUT_DIR="None"
resume=False
PROMPT_PATTERN="fixbycrflp"
SAVE_FILE="fixbycrflp2_WsAndS_Part"
DATA_FILE="NewPartDataset"
num_train_epochs=10
train_batch_size=4
eval_batch_size=2
test_batch_size=1
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
    # 构建文件路径
    file_path="./output_dir/loraWeight/$SAVE_FILE"
    # 检查文件是否存在
    if [ -e "$file_path" ]; then
        echo "文件 $file_path 存在"
    else
        echo "目录 $file_path 不存在，创建目录"
        mkdir -p "$file_path"
    fi

    echo ">>>start Train"
    #bash ./script/pipeline/part/train.sh "$gpu_seq" "$PROMPT_PATTERN" "$DATA_FILE" "$PORT" "$gpu_count" "$resume" "$num_train_epochs" "$newloss_pattern" "$debug_mode"
    export NCCL_P2P_DISABLE=1
    export CUDA_LAUNCH_BLOCKING=1
    export TORCH_USE_CUDA_DSA=1
    CUDA_VISIBLE_DEVICES=$gpu_seq torchrun --master_port=$PORT --nproc_per_node $gpu_count -m LoraTrainer.train \
        --do_train True \
        --do_eval True \
        --debug_mode $debug_mode \
        --bf16 True \
        --prompt_pattern "$PROMPT_PATTERN" \
        --input_dir "$INPUT_DIR"\
        --output_dir "./output_dir/loraWeight/$SAVE_FILE/" \
        --resume $resume\
        --num_train_epochs $num_train_epochs \
        --per_device_train_batch_size $train_batch_size \
        --per_device_eval_batch_size $train_batch_size \
        --save_steps 2000 \
        --eval_steps 10000 \
        --logging_steps 50 \
        --data_path "./repairDataset/RepairData-PythonLevel/$DATA_FILE/" \
        --deepspeed configs/default_offload_opt_param-o2.json \
        --save_safetensors False \
        --evaluation_strategy steps\
        --save_strategy steps \
        --remove_unused_columns False \

fi



if [ "$dev_status" = "true" ]; then

    # 构建文件路径
    file_path="./predict_dir/loraWeight/$SAVE_FILE"
    # 检查文件是否存在
    if [ -e "$file_path" ]; then
        echo "文件 $file_path 存在"
    else
        echo "目录 $file_path 不存在，创建目录"
        mkdir -p "$file_path"
    fi

    echo ">>>start Eval DEV to gengeration"
    bash ./script/pipeline/part/recyle_eval2.sh "$end_iterations" "$PATTERN" "$begin_iterations" "$SAVE_STEPS" "$PROMPT_PATTERN" "$DATA_FILE" "$gpu_seq2eval" "$DEVPORT"\
     "$eval_batch_size" "$SAVE_FILE" "False" "None"
fi


if [ "$Exec_status" = "true" ]; then
    # 构建文件路径
    file_path="./predict_evalResult_dir/$SAVE_FILE"
    # 检查文件是否存在
    if [ -e "$file_path" ]; then
        echo "文件 $file_path 存在"
    else
        echo "目录 $file_path 不存在，创建目录"
        mkdir -p "$file_path"
    fi

    echo ">>>start Execution" 
    bash ./script/pipeline/part/recyle_Execution.sh "$end_iterations" "$PATTERN" "20000" "$SAVE_STEPS" "$SAVE_FILE" #$begin_iterations
fi

if [ "$statistic_status" = "true" ]; then
    # 初始化最大 rate 和对应的 i
    max_rate=0.0
    best_iteration=0
    echo ">>>start choose best dev rate value"
    # 遍历迭代范围，执行脚本并捕获输出
    for ((i=begin_iterations; i<=end_iterations; i+=SAVE_STEPS)); do
        output=$(bash ./script/pipeline/part/Statistical_Execution_Results.sh "$SAVE_FILE" "$PATTERN" "$i")
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
    bash ./script/pipeline/part/recyle_eval2.sh "$best_iteration" "test" "$best_iteration" "$SAVE_STEPS" "$PROMPT_PATTERN" "$DATA_FILE" "$gpu_seq2test" "$DEVPORT"\
     "$test_batch_size" "$SAVE_FILE" "True" "$test_CRPdata_path"
    echo ">>>start Execution For Test" 
    bash ./script/pipeline/part/recyle_Execution.sh "$best_iteration" "test" "$best_iteration" "$SAVE_STEPS" "$SAVE_FILE"
    echo ">>>start Statistic Execution Result For test"
    bash ./script/pipeline/part/Statistical_Execution_Results.sh "$SAVE_FILE" "test" "$best_iteration"
fi