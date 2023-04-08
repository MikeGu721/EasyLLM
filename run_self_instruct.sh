python generate-instructions.py generate_instruction_following_data \
    --output_dir=./instruction_data \
    --seed_tasks_path=./instruction_data/zh_seed_tasks.json \
    --request_batch_size=2 \
    --api=chat \
    --num_cpus=1 \
    --model_name=gpt35
