export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
CUDA_VISIBLE_DEVICES=$1 torchrun --master_port=$4 --nproc_per_node $5 -m LoraTrainer.train \
    --do_train True \
    --do_eval True \
    --debug_mode $9\
    --bf16 True \
    --prompt_pattern "$2" \
    --output_dir "./output_dir/loraWeight/$2/" \
    --resume $6\
    --num_train_epochs $7 \
    --per_device_train_batch_size ${11} \
    --per_device_eval_batch_size ${11} \
    --save_steps 2000 \
    --eval_steps ${10} \
    --logging_steps 50 \
    --data_path "./repairDataset/RepairData-PythonLevel/$3/" \
    --deepspeed configs/default_offload_opt_param-o2.json \
    --save_safetensors False \
    --evaluation_strategy steps\
    --save_strategy steps \
    --remove_unused_columns False \
    --newloss_pattern $8
