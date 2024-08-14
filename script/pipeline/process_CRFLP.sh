#!/bin/bash

# 从命令行参数获取循环次数
begin_iterations=2000
end_iterations=30000
PORT=61200
DEVPORT=$((PORT + 10))
gpu_seq=4,5,6,7
gpu_seq2eval=4,5,6,7
gpu_seq2test=4,5,6,7
PATTERN="dev" 
SAVE_STEPS=2000
eval_steps=6000
PROMPT_PATTERN="CRFLP"
SECOND_PROMPT_PATTERN="fixbycrflp"
second_checkpoint=8000
DATA_FILE="CRFLPDataset"
resume=False
num_train_epochs=10
train_batch_size=4
eval_batch_size=3
test_batch_size=1
newloss_pattern=False #改成True
debug_mode=False
# 获取 GPU 个数
IFS=',' read -ra GPU_ARRAY <<< "$gpu_seq"
gpu_count=${#GPU_ARRAY[@]}


echo ">>>start Train"
bash ./script/pipeline/part/train.sh "$gpu_seq" "$PROMPT_PATTERN" "$DATA_FILE" "$PORT" "$gpu_count" "$resume" "$num_train_epochs" "$newloss_pattern" "$debug_mode" "$eval_steps" "$train_batch_size"


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

 # 构建文件路径
file_path="./predict_dir/loraWeight/$PROMPT_PATTERN-GEN"
# 检查文件是否存在
if [ -e "$file_path" ]; then
    echo "文件 $file_path 存在"
else
    echo "目录 $file_path 不存在，创建目录"
    mkdir -p "$file_path"
fi

echo ">>>start eval dev to fix buggy code by fixModel"
bash ./script/pipeline/part/recyle_choseCRP.sh "$end_iterations" "$PATTERN" "$begin_iterations" "$SAVE_STEPS" "$PROMPT_PATTERN" "$DATA_FILE" "$gpu_seq2eval" "$DEVPORT" "$eval_batch_size" \
"$SECOND_PROMPT_PATTERN" "$second_checkpoint"


# 构建文件路径
file_path="./predict_evalResult_dir/$PROMPT_PATTERN-GEN"
# 检查文件是否存在
if [ -e "$file_path" ]; then
    echo "文件 $file_path 存在"
else
    echo "目录 $file_path 不存在，创建目录"
    mkdir -p "$file_path"
fi
echo ">>>start Execution" 
bash ./script/pipeline/part/recyle_Execution.sh "$end_iterations" "$PATTERN" "$begin_iterations" "$SAVE_STEPS" "$PROMPT_PATTERN-GEN"

#需要通过genbycpr执行得到结果


#初始化最大 rate 和对应的 i
max_rate=0.0
best_iteration=0
echo ">>>start choose best dev rate value"
# 遍历迭代范围，执行脚本并捕获输出
for ((i=begin_iterations; i<=end_iterations; i+=SAVE_STEPS)); do
    output=$(bash ./script/pipeline/part/Statistical_Execution_Results.sh "$PROMPT_PATTERN-GEN" "$PATTERN" "$i")
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


echo ">>>start Eval TEST to gengeration  $best_iteration"
#test:第一阶段生成crp
bash ./script/pipeline/part/recyle_eval.sh "$best_iteration" "test" "$best_iteration" "$SAVE_STEPS" "$PROMPT_PATTERN" "$DATA_FILE" "$gpu_seq2test" "$DEVPORT" "$test_batch_size"
#test:第二阶段基于crp执行
bash ./script/pipeline/part/recyle_choseCRP.sh "$best_iteration" "test" "$best_iteration" "$SAVE_STEPS" "$PROMPT_PATTERN" "$DATA_FILE" "$gpu_seq2test" "$DEVPORT" "$test_batch_size" \
"$SECOND_PROMPT_PATTERN" "$second_checkpoint"
echo ">>>start Execution For TESt" 
bash ./script/pipeline/part/recyle_Execution.sh "$best_iteration" "test" "$best_iteration" "$SAVE_STEPS" "$PROMPT_PATTERN-GEN"
echo ">>>start Statistic Execution Result For TEST"
bash ./script/pipeline/part/Statistical_Execution_Results.sh "$PROMPT_PATTERN-GEN" "test" "$best_iteration"