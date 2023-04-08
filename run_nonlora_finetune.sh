export BASE_MODEL=decapoda-research/llama-7b-hf
#export BASE_MODEL=THUDM/glm-10b-chinese
#export BASE_MODEL=bigscience/bloomz-7b1
#export BASE_MODEL=bigscience/bloomz-560m

export DATA_PATH=./data/instruction_data/ift.data.json
export DATA_NAME=sft-data
export CACHE_DIR=./model/download_model

export OUTPUT_DIR=./model/ifted_model

export NUM_GPUS=3

deepspeed --NUM_GPUS=8 nonlora-finetune.py \
    --base_model $BASE_MODEL \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR'/'$DATA_NAME'/'$BASE_MODEL \
    --batch_size 128 \
    --micro_batch_size 4 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --group_by_length \
    --cache_dir $CACHE_DIR \
    --deepspeed ds_config.json
