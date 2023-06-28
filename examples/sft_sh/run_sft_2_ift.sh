export SFT_FILE='data/instruction_data/new_ift_data.not_cleaned.json'
export IFT_FILE='data/instruction_data/ift.data.json'

python data/instruction_data/sft2ift.py \
        --sft_file=$SFT_FILE \
        --ift_file=$IFT_FILE
