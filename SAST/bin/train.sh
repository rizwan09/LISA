#!/usr/bin/env bash

config_file=$1

source ${config_file}

params=${@:2}

echo "Using CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES

transition_stats=$data_dir/transition_probs.tsv

python3 src/train.py \
--train_files $train_files \
--dev_files $dev_files \
--transition_stats $transition_stats \
--data_config $data_config \
--model_configs $model_configs \
--task_configs $task_configs \
--layer_configs $layer_configs \
--attention_configs "$attention_configs" \
--dev_parse_files $dev_parse_files \
--train_parse_files $train_parse_files \
--best_eval_key $best_eval_key \
$params

