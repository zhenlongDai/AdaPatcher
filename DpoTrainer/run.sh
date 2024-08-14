export CUDA_VISIBLE_DEVICES=0,1,2,5
ACCELERATE_LOG_LEVEL=info accelerate launch -m DpoTrainer.dpo_llama \
    --model_name_or_path="./output_dir/sft_codeLlama" \
    --output_dir="./output_dir/DpoWeight/V3"
#--deepspeed configs/default_offload_opt_param-o2.json 
