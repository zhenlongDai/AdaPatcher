export CUDA_VISIBLE_DEVICES=$1
ACCELERATE_LOG_LEVEL=info accelerate launch -m DpoTrainer.dpo_llama \
    --model_name_or_path="./output_dir/loraWeight/trace_CRFLP/checkpoint-14000" \
    --output_dir="./output_dir/DpoWeight/$2"\
    --save_steps $3 \
    --num_train_epochs $4 \
    --per_device_train_batch_size $5 \
    --per_device_eval_batch_size $5 
#--deepspeed configs/default_offload_opt_param-o2.json 
