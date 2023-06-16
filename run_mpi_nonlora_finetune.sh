export BASE_MODEL=decapoda-research/llama-13b-hf

export DATA_PATH=./data/instruction_data/ift.data.json
export DATA_NAME=sft-data
export CACHE_DIR=NONE

export OUTPUT_DIR=./model/ifted_model
export DEEPSPEED_STAGE=3

DISTRIBUTED_ARGS="\
        --nproc_per_node=$GPU_NUM \
        --nnodes=$WORLD_SIZE \
        --node_rank=$RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        "

python -m torch.distributed.launch $DISTRIBUTED_ARGS nonlora-finetune.py \
    --base_model $BASE_MODEL \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR'/'$DATA_NAME'/'$BASE_MODEL \
    --batch_size 128 \
    --micro_batch_size 3 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --group_by_length \
    --cache_dir $CACHE_DIR \
    --deepspeed_stage $DEEPSPEED_STAGE