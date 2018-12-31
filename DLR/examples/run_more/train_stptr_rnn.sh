#!/usr/bin/env bash

# train & test

# mkdir tmp model; cp -r ../run_dict/model/alphabets/ ./model/;

RGPU=$1
SEED=$2
MODEL_NAME=$3
MODEL_DIR='./model2.2/'

EMB_DIR=~/zhisong/zr0920_final/data2.2_more/
DATA_DIR=~/zhisong/zr0920_final/data2.2_more/
SRC_DIR=../src/

EN_TRAIN="${DATA_DIR}/en_train.conllu"     # English train path
EN_DEV="${DATA_DIR}/en_dev.conllu"       # English dev path
EN_TEST="${DATA_DIR}/en_test.conllu"       # English test path

#mkdir $MODEL_DIR tmp
#PYTHONPATH=${SRC_DIR} CUDA_VISIBLE_DEVICES= python2 ${SRC_DIR}/examples/vocab/build_joint_vocab_embed.py \
#    --embed_paths ${EMB_DIR}/wiki.multi.en.vec \
#    --embed_lang_ids en \
#    --data_paths $EN_TRAIN $EN_DEV \
#    --model_path $MODEL_DIR

echo "Current seed is $SEED"

PYTHONPATH=../src/ CUDA_VISIBLE_DEVICES=$RGPU python2 ${SRC_DIR}/examples/StackPointerParser.py \
--mode FastLSTM \
--self_attn \
--decoder_input_size 256 \
--hidden_size 300 \
--encoder_layers 3 \
--d_k 64 \
--d_v 64 \
--decoder_layers 1 \
--arc_space 512 \
--type_space 128 \
--opt adam \
--decay_rate 0.75 \
--epsilon 1e-4 \
--coverage 0.0 \
--gamma 0.0 \
--clip 5.0 \
--schedule 20 \
--double_schedule_decay 5 \
--use_warmup_schedule \
--check_dev 5 \
--unk_replace 0.5 \
--label_smooth 1.0 \
--beam 1 \
--freeze \
--pos \
--pool_type weight \
--multi_head_attn \
--num_head 8 \
--word_embedding word2vec \
--word_path ${MODEL_DIR}/alphabets/joint_embed.vec \
--char_embedding random \
--punctuation 'PUNCT' 'SYM' \
--train $EN_TRAIN \
--dev $EN_DEV \
--test $EN_TEST \
--vocab_path $MODEL_DIR \
--model_path $MODEL_DIR \
--model_name $MODEL_NAME \
--p_in 0.33 \
--p_out 0.33 \
--p_rnn 0.33 0.33 \
--learning_rate 0.001 \
--num_epochs 1000 \
--trans_hid_size 512 \
--pos_dim 50 \
--char_dim 50 \
--num_filters 50 \
--prior_order left2right \
--grandPar \
--batch_size 32 \
--seed $SEED

#RGPU=$RGPU bash -v ../src/examples/run_more/run_analyze.sh dev stackptr |& tee log_dev
# RGPU=$RGPU bash -v ../src/examples/run_more/run_analyze.sh test stackptr |& tee log_test
# RGPU=$RGPU bash -v ../src/examples/run_more/run_analyze.sh train stackptr |& tee log_train

#
# b neuronlp2/transformer/multi_head_attn:104
# b neuronlp2/models/parsing:438

# run
# RGPU=2 bash -v go.sh |& tee log
