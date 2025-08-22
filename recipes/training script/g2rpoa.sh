ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes=7 src/open_r1/grpo_code_adagui.py --config recipes/Qwen3-1.7B/grpo/qwen38code.yaml
