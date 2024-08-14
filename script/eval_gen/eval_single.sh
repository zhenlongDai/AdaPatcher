export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
CUDA_VISIBLE_DEVICES=4 torchrun --master_port=63601 --nproc_per_node 1 -m LoraTrainer.eval_single \
    --do_eval True \
    --debug_mode False\
    --use_peft True \
    --output_dir "./output_dir/loraWeight/normal/checkpoint-24000/" \
    --predict_dir "./predict_dir/loraWeight/6-11/"\
    --per_device_eval_batch_size 2 \
    --save_steps 200 \
