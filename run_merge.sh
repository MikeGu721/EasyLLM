

python merge_weights.py \
      --BASE_MODEL=bigscience/bloomz-560m \
      --cache_dir=./model/download_model \
      --lora_weights=./model/ifted_model/sft-data/bigscience/bloomz-560m \
      --save_dir=./model/save_models \
      --load_in_8bit=False \
      --max_shard_size=400MB

