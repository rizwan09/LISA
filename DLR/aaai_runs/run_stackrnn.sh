#!/usr/bin/env bash

# !!: specify the paths
SRC_DIR=~/zhisong/zr0920_final/src/
EMB_DIR=~/zhisong/zr0920_final/data2.2_more/

RGPU=$1         # which GPU
DATA_DIR="data2.0/"     # where are the target data: name should be ${lang}_test.conllu

EN_TRAIN="${DATA_DIR}/en_train.conllu"     # English train path
EN_DEV="${DATA_DIR}/en_dev.conllu"       # English dev path

# step 1: prepare the vocab
# mkdir model tmp

#PYTHONPATH=${SRC_DIR} CUDA_VISIBLE_DEVICES= python2 ${SRC_DIR}/examples/vocab/build_joint_vocab_embed.py \
#    --embed_paths ${EMB_DIR}/wiki.multi.en.vec \
#    --embed_lang_ids en \
#    --data_paths $EN_TRAIN $EN_DEV \
#    --model_path ./model/

# step 2: train the English parser (here only 500 epochs to save time)
PYTHONPATH=${SRC_DIR} CUDA_VISIBLE_DEVICES=$RGPU python2 ${SRC_DIR}/examples/StackPointerParser.py \
    --mode FastLSTM \
    --hidden_size 300 \
    --encoder_layers 3 \
    --d_k 64 \
    --d_v 64 \
    --decoder_input_size 256 \
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
    --word_path './model/alphabets/joint_embed.vec' \
    --char_embedding random \
    --punctuation 'PUNCT' 'SYM' '.' \
    --train $EN_TRAIN \
    --dev $EN_DEV \
    --test $EN_DEV \
    --vocab_path './model/' \
    --model_path './model/' \
    --model_name 'stack_rnn.pt' \
    --p_in 0.33 \
    --p_out 0.33 \
    --p_rnn 0.33 0.33 \
    --learning_rate 0.001 \
    --num_epochs 500 \
    --trans_hid_size 512 \
    --pos_dim 50 \
    --char_dim 50 \
    --num_filters 50 \
    --input_concat_embeds \
    --input_concat_position \
    --position_dim 0 \
    --prior_order left2right \
    --grandPar \
    --enc_clip_dist 0 \
    --batch_size 32 \
    --seed 1234

# step 3: test on all target languages
# for cur_lang in de es fr pt sv it;
for cur_lang in de es fr;
do
        PYTHONPATH=${SRC_DIR} CUDA_VISIBLE_DEVICES=$RGPU python2 ${SRC_DIR}/examples/analyze.py --parser biaffine --ordered --gpu\
                    --punctuation 'PUNCT' 'SYM' '.' --out_filename analyzer.${cur_lang}.out --model_name 'stack_rnn.pt' \
                            --test "${DATA_DIR}/${cur_lang}_test.conllu" --model_path "./model/" --extra_embed "${EMB_DIR}/wiki.multi.${cur_lang}.vec"
    done

