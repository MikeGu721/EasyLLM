export MEGA_CPT_FILE=./model/pretrained_model/test_weight/test_weight.pt
export CONFIG_FILE=./model/pretrained_model/test_weight/config.json


python convert_megatron_2_hf.py \
        --path_to_checkpoint $MEGA_CPT_FILE \
        --config_file $CONFIG_FILE \
        --print-checkpoint-structure
