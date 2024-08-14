export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=3,7
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/deepspeed_zero3.yaml DpoTrainer/run_simpo.py configs/LoraCodeLlama-instruct-simpo.yaml