

PATTERN="test"
DATA_FILE="CRFLPDataset"
gpu_seq=4,5,6,7
PORT=37000
batch_size=1
PROMPT_PATTERN="fixbycrflp"
predict_filePath="predict_dir/DPoWeight/test/FixCodeLlamaFixResult.json"
use_predict_crp=true
test_CRPdata_path="./predict_dir/loraWeight/trace_CRFLP/test-checkpoint-14000.json"
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
    --output_dir "./output_dir/fix_codeLlama/" \
    --data_path "./repairDataset/RepairData-PythonLevel/$DATA_FILE/" \
    --predict_filePath "$predict_filePath" \
    --per_device_eval_batch_size $batch_size \
    --use_predict_crp "$use_predict_crp" \
    --CRPdata_path "$test_CRPdata_path"
echo "All evaluations completed."
