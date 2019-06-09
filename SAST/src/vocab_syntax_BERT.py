import os
run_command = 'python /home/rizwan/SBCR/tensor2tensor/tensor2tensor/data_generators/text_encoder_build_subword.py \
    --corpus_filepattern=/home/rizwan/SBCR/SAST/SyntaxCorpus/vocab/dummy.txt\
    --corpus_max_lines=1000000000000000\
    --output_filename=/home/rizwan/SBCR/SAST/SyntaxCorpus/vocab/log.txt \
    --logtostderr'

os.system(run_command)
