# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Pre-train fac-adapter

task=trex
GPU='0,1,2,3'
CUDA_VISIBLE_DEVICES=$GPU python fac-adapter.py  \
        --model_type roberta \
        --model_name=roberta-large  \
        --data_dir=./data/trex-rc  \
        --output_dir trex_output \
        --restore '' \
        --do_train  \
        --do_eval   \
        --evaluate_during_training 'True' \
        --task_name=$task     \
        --comment 'fac-adapter' \
        --per_gpu_train_batch_size=64   \
        --per_gpu_eval_batch_size=64   \
        --num_train_epochs 5 \
        --max_seq_lengt 64 \
        --gradient_accumulation_steps 1 \
        --learning_rate 5e-5 \
        --warmup_steps=1200 \
        --save_steps 1000 \
        --adapter_size 768 \
        --adapter_list "0,11,22" \
        --adapter_skip_layers 0 \
        --adapter_transformer_layers 2 \
        --meta_adapter_model=""