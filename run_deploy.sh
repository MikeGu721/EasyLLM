#export BASE_MODEL=decapoda-research/llama-7b-hf
#export BASE_MODEL=THUDM/glm-10b-chinese
#export BASE_MODEL=bigscience/bloomz-7b1
#export BASE_MODEL=bigscience/bloomz-560m
export BASE_MODEL=./model/save_models/hf_ckpt

export DATA_NAME=sft-data
#export CACHE_DIR=./model/download_model
export CACHE_DIR=NONE
export OUTPUT_DIR=NONE
#export OUTPUT_DIR=./model/ifted_model


python deploy.py \
    --base_model $BASE_MODEL \
    --cache_dir $CACHE_DIR \
    --lora_weights $OUTPUT_DIR'/'$DATA_NAME'/'$BASE_MODEL \
    --prompt_templare gzh_prompter \
    --server_name 10.176.64.186
