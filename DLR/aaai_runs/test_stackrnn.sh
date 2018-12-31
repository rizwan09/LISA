#!/usr/bin/env bash

# !!: specify the paths
SRC_DIR=~/zhisong/zr0920_final/src/
EMB_DIR=~/zhisong/zr0920_final/data2.2_more/

RGPU=$1         # which GPU
DATA_DIR="data2.0/"     # where are the target data: name should be ${lang}_test.conllu

# step 3: test on all target languages
# for cur_lang in de es fr pt sv it;
for cur_lang in de es fr;
do
        PYTHONPATH=${SRC_DIR} CUDA_VISIBLE_DEVICES=$RGPU python2 ${SRC_DIR}/examples/analyze.py \
                    --parser stackptr --beam 5 --ordered --gpu \
                    --punctuation 'PUNCT' 'SYM' '.' \
                    --out_filename analyzer.${cur_lang}.out \
                    --model_name 'stack_rnn.pt' \
                    --test "${DATA_DIR}/${cur_lang}_test.conllu" \
                    --model_path "./model/" \--extra_embed "${EMB_DIR}/wiki.multi.${cur_lang}.vec"
    done

