import os
import sys
import fire
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers.models.llama import LlamaTokenizer, LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BloomForCausalLM
from typing import List
from prompter_setting.Prompter import Prompter

# os.environ['CURL_CA_BUNDLE'] = ''


def start_ift(base_model: str,
              data_path: str,
              output_dir: str,
              batch_size: int = 128,
              micro_batch_size: int = 4,
              num_epochs: int = 3,
              learning_rate: float = 3e-4,
              cutoff_len: int = 512,
              val_set_size: int = 2000,
              lora_r: int = 8,
              lora_alpha: int = 16,
              lora_dropout: float = 0.05,
              lora_target_modules: List[str] = ['q_proj', 'v_proj'],
              train_on_inputs: bool = False,
              group_by_length: bool = False,
              wandb_project: str = '',
              wandb_run_name: str = '',
              wandb_watch: str = '',
              wandb_log_model: str = '',
              resume_from_checkpoint: str = None,  # 需要读取参数的目录，而不是精确到文件
              prompt_template_name: str = 'gzh_prompter',
              cache_dir: str = '.' ):
    print('开始进行显卡设置')
    # 一些读取
    prompter = Prompter(prompt_template_name)
    gradient_accumulation_steps = batch_size // micro_batch_size
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))  # 获得总进程数
    ddp = world_size != 1  # 如果进程数不为1
    if ddp:  # 需要多卡并行
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}  # 获得本卡的地址
        gradient_accumulation_steps = gradient_accumulation_steps // world_size  # 梯度累计再除以总进程数
    # ——————————————————————————————模型训练过程记录——————————————————————————————
    use_wandb = len(wandb_project) > 0 or ("WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0)  #
    if len(wandb_project) > 0: os.environ['WANDB_PROJECT'] = wandb_project
    if len(wandb_watch) > 0: os.environ['WANDB_WATCH'] = wandb_watch
    if len(wandb_log_model) > 0: os.environ['WANDB_LOG_MODEL'] = wandb_log_model
    # ——————————————————————————————模型训练过程记录——————————————————————————————
    print('显卡设置完毕')

    # 设置tokenizer
    if 'llama' in base_model.lower():
        tokenizer = LlamaTokenizer.from_pretrained(base_model, cache_dir=cache_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=cache_dir)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'left'

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(prompt, truncation=True, max_length=cutoff_len, padding=False, return_tensors=None)
        if (result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point['instruction'],
            data_point['input'],
            data_point['output']
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:  # input内容不产生loss
            user_prompt = prompter.generate_prompt(data_point['instruction'], data_point['input'])
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt['input_ids'])

            tokenized_full_prompt['labels'] = [-100] * user_prompt_len + tokenized_full_prompt['labels'][
                                                                         user_prompt_len:]
        return tokenized_full_prompt

    print('完成Tokenizer设置')


    # 数据处理
    data = load_dataset("json", data_files=data_path)
    train_val = data['train'].train_test_split(test_size=val_set_size, shuffle=True, seed=42)
    if val_set_size > 0:
        train_data = train_val['train'].shuffle().map(generate_tokenize_prompt)
        val_data = train_val['test'].shuffle().map(generate_tokenize_prompt)
    else:
        train_data = train_val['train'].shuffle().map(generate_tokenize_prompt)
        val_data = None
    print('数据处理完成，数据大小：', sys.getsizeof(train_val))

    # 准备模型
    if 'llama' in base_model.lower():
        model = LlamaForCausalLM.from_pretrained(base_model, cache_dir=cache_dir, load_in_8bit=False,
                                                 device_map=device_map, )
        print('Using Llama')
    elif 'bloom' in base_model.lower():
        model = BloomForCausalLM.from_pretrained(base_model, cache_dir=cache_dir, load_in_8bit=False,
                                                 device_map=device_map, )
        print('Using Bloom')
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model, cache_dir=cache_dir, load_in_8bit=False,
                                                     device_map=device_map, )
    print(model)
    # model = PeftModel.from_pretrained(model, PEFT_MODEL)
    # model = prepare_model_for_int8_training(model)  # 和 from_pretrained 8bt的参数保持一致
    config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=lora_target_modules, lora_dropout=lora_dropout,
                        bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, config)
    # 模型并行
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    print('模型准备完毕')

    # 读取之前的模型
    if resume_from_checkpoint:
        checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")
        if not os.path.exists(checkpoint_name):  # 只读取旁路的参数
            checkpoint_name = os.path.join(resume_from_checkpoint, "adapter_model.bin")
            resume_from_checkpoint = False
        if os.path.exists(checkpoint_name):
            print(f"从{checkpoint_name}继续开始ift")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} 地址不存在")
        print(f'已读取{checkpoint_name}的模型')
    model.print_trainable_parameters()
    # 设置Trainer
    transformer_args = transformers.TrainingArguments(per_device_train_batch_size=micro_batch_size,
                                                      gradient_accumulation_steps=gradient_accumulation_steps,
                                                      warmup_steps=100,
                                                      num_train_epochs=num_epochs,
                                                      learning_rate=learning_rate,
                                                      fp16=True,
                                                      logging_steps=10,
                                                      optim='adamw_torch',
                                                      evaluation_strategy='steps' if val_set_size > 0 else 'no',
                                                      save_strategy='steps',
                                                      eval_steps=200 if val_set_size > 0 else None,
                                                      save_steps=200,
                                                      output_dir=output_dir,
                                                      save_total_limit=3,
                                                      load_best_model_at_end=True if val_set_size > 0 else False,
                                                      ddp_find_unused_parameters=False if ddp else None,
                                                      group_by_length=group_by_length,
                                                      report_to='wandb' if use_wandb else None,
                                                      run_name=wandb_run_name if use_wandb else None)
    transformer_collator = transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors='pt',
                                                               padding=True)
    trainer = transformers.Trainer(model=model, train_dataset=train_data, eval_dataset=val_data, args=transformer_args,
                                   data_collator=transformer_collator)
    model.config.use_cache = False
    old_state_dict = model.state_dict
    model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(model,
                                                                                                          type(model))
    print('完成Trainer设置')

    # Start Training
    print('开始训练模型：')
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print('模型训练完成')

    # 保存模型参数
    model.save_pretrained(output_dir)
    print(f'已保存模型参数至：{output_dir}')


if __name__ == '__main__':
    fire.Fire(start_ift)
