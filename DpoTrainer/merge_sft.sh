export CUDA_VISIBLE_DEVICES=4,5,6,7
ACCELERATE_LOG_LEVEL=info accelerate launch -m DpoTrainer.merge_sft \
    --model_name_or_path="./output_dir/loraWeight/fixbycrflp2/checkpoint-12000"\
    --output_name="./output_dir/fix_codeLlama"
#--deepspeed configs/default_offload_opt_param-o2.json 
