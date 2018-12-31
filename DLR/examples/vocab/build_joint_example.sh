#!/usr/bin/env bash

# build

EMB_DIR=~/zhisong/zr0920_final/data2.2_more/
DATA_DIR=~/zhisong/zr0920_final/data2.2_more/
SRC_DIR=../src/

embed_path1=${EMB_DIR}/wiki.multi.en.vec
embed_path2=${EMB_DIR}/wiki.multi.de.vec
embed_path3=${EMB_DIR}/wiki.multi.fr.vec

data0=${DATA_DIR}/en_train.conllu
data1=${DATA_DIR}/en_dev.conllu
data2=${DATA_DIR}/en_test.conllu

data3=${DATA_DIR}/de_train.conllu
data4=${DATA_DIR}/de_dev.conllu
data5=${DATA_DIR}/de_test.conllu

data6=${DATA_DIR}/fr_train.conllu
data7=${DATA_DIR}/fr_dev.conllu
data8=${DATA_DIR}/fr_test.conllu

PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES= python2 ${SRC_DIR}/examples/vocab/build_joint_vocab_embed.py \
    --embed_paths $embed_path1 $embed_path2 $embed_path3 \
	--embed_lang_ids en de fr \
	--data_paths $data0 $data1 $data2 $data3 $data4 $data5 $data6 $data7 $data8 \
	--model_path ./model_en_de_fr/
