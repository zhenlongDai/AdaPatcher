export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --master_port=61000 --nproc_per_node 4 -m LoraTrainer.train \
    --do_train True \
    --do_eval True \
    --debug_mode False\
    --bf16 True \
    --prompt_pattern "normal" \
    --output_dir "./output_dir/loraWeight/normal/" \
    --resume False\
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --save_steps 2000 \
    --eval_steps 10000 \
    --logging_steps 50 \
    --data_path "./repairDataset/RepairData-PythonLevel/traceDataset/" \
    --deepspeed configs/default_offload_opt_param-o3.json \
    --save_safetensors False \
    --evaluation_strategy steps\
    --save_strategy steps \
    --remove_unused_columns False 
