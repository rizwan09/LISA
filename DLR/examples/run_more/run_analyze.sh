#!/usr/bin/env bash

MODEL_NAME='stptr_rnn_attn.pt'
MODEL_DIR='./model2.2/'
EMB_DIR=~/zhisong/zr0920_final/data2.2_more/
DATA_DIR=~/zhisong/zr0920_final/data2.2_more/
SRC_DIR=../src/

#
function run_lang () {

echo "======================"
echo "Running with lang = $1_$2_$3"

cur_lang=$1
which_set=$2
which_model=$3

# try them both, will fail on one

if [ "$which_model" == "biaffine" ]; then

echo PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES=$RGPU python2 ${SRC_DIR}/examples/analyze.py --parser biaffine --ordered --gpu \
--punctuation 'PUNCT' 'SYM' --out_filename analyzer.$cur_lang.$which_set.out --model_name $MODEL_NAME \
--test "${DATA_DIR}/${cur_lang}_${which_set}.conllu" --model_path $MODEL_DIR --extra_embed "${EMB_DIR}/wiki.multi.${cur_lang}.vec"

PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES=$RGPU python2 ${SRC_DIR}/examples/analyze.py --parser biaffine --ordered --gpu \
--punctuation 'PUNCT' 'SYM' --out_filename analyzer.$cur_lang.$which_set.out --model_name $MODEL_NAME \
--test "${DATA_DIR}/${cur_lang}_${which_set}.conllu" --model_path $MODEL_DIR --extra_embed "${EMB_DIR}/wiki.multi.${cur_lang}.vec"

elif [ "$which_model" == "stackptr" ]; then

echo PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES=$RGPU python2 ${SRC_DIR}/examples/analyze.py --parser stackptr --beam 5 --ordered --gpu \
--punctuation 'PUNCT' 'SYM' --out_filename analyzer.$cur_lang.$which_set.out --model_name $MODEL_NAME \
--test "${DATA_DIR}/${cur_lang}_${which_set}.conllu" --model_path $MODEL_DIR --extra_embed "${EMB_DIR}/wiki.multi.${cur_lang}.vec"

PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES=$RGPU python2 ${SRC_DIR}/examples/analyze.py --parser stackptr --beam 5 --ordered --gpu \
--punctuation 'PUNCT' 'SYM' --out_filename analyzer.$cur_lang.$which_set.out --model_name $MODEL_NAME \
--test "${DATA_DIR}/${cur_lang}_${which_set}.conllu" --model_path $MODEL_DIR --extra_embed "${EMB_DIR}/wiki.multi.${cur_lang}.vec"

fi

}

# =====

echo "Run them all with $1_$2"

# running with which dev, which set?
for cur_lang in ar bg ca zh hr cs da nl en et "fi" fr de he hi id it ja ko la lv no pl pt ro ru sk sl es sv uk;
do
    run_lang $cur_lang $1 $2;
done

# RGPU=0 bash ../src/examples/run_more/run_analyze.sh |& tee log_test
# see the results?
# -> cat log_test | grep -E "python2|Running with lang|uas|Error"
# -> cat log_test | grep -E "Running with lang|uas"
# -> cat log_test | grep -E "Running with lang|test Wo Punct"
# -> cat log_test | python3 print_log_test.py
