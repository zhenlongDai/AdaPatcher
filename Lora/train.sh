export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
CUDA_VISIBLE_DEVICES=4,5 torchrun --master_port=61666 --nproc_per_node 2 -m Lora.train \
    --do_train True \
    --do_eval True \
    --debug_mode True\
    --output_dir "./output_dir/loraWeight/" \
    --resume False\
    --use_peft False \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2\
    --num_train_epochs 100 \
    --deepspeed "configs/default_offload_opt_param.json" \
