export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port=63700 --nproc_per_node 2 -m LoraTrainer.eval_Multi \
    --do_eval True \
    --debug_mode False\
    --use_peft True \
    --eval_pattern "test" \
    --output_dir "./output_dir/loraWeight/6-11/checkpoint-7400/" \
    --predict_filePath  "./predict_dir/loraWeight/6-11/checkpoint-7400_test.json" \
    --per_device_eval_batch_size 2 \
    --save_steps 200 \
