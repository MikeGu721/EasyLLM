#! /bin/bash
# Runs the "345M" parameter model

RANK=0
WORLD_SIZE=1


VOCAB_FILE=../model/gpt2-setting/gpt2-vocab.json
MERGE_FILE=../model/gpt2-setting/gpt2-merge.txt
RAW_DATA_PATH=../data/pretrain_data/test_data.json
DATA_PATH=my-gpt2_text_document
CHECKPOINT_PATH=../model/pretrained_model

cd Megatron-DeepSpeed
python tools/preprocess_data.py \
       --input $RAW_DATA_PATH \
       --output-prefix my-gpt2 \
       --vocab $VOCAB_FILE \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file $MERGE_FILE \
       --append-eod

python pretrain_gpt.py \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 4 \
       --global-batch-size 16 \
       --seq-length 8 \
       --max-position-embeddings 1024 \
       --train-iters 500000 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --min-lr 1.0e-5 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --checkpoint-activations \
       --log-interval 100 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16