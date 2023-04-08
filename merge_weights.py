import os

import fire
import torch
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer, BloomForCausalLM, AutoTokenizer, AutoModelForCausalLM


def export_hf(BASE_MODEL: str = '',
              cache_dir: str = '',
              lora_weights: str = '',
              save_dir: str = '',
              load_in_8bit: bool = False,
              max_shard_size: str = "400MB"):
    # 读取模型
    if 'llama' in BASE_MODEL.lower():
        base_model = LlamaForCausalLM.from_pretrained(BASE_MODEL, cache_dir=cache_dir, load_in_8bit=load_in_8bit,
                                                      torch_dtype=torch.float16, device_map={"": "cpu"}, )
    elif 'bloom' in BASE_MODEL.lower():
        base_model = BloomForCausalLM.from_pretrained(BASE_MODEL, cache_dir=cache_dir, load_in_8bit=load_in_8bit,
                                                      torch_dtype=torch.float16, device_map={"": "cpu"}, )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, cache_dir=cache_dir, load_in_8bit=load_in_8bit,
                                                          torch_dtype=torch.float16, device_map={"": "cpu"}, )
    lora_model = PeftModel.from_pretrained(base_model, lora_weights, device_map={"": "cpu"},
                                           torch_dtype=torch.float16, )

    # 读取Tokenizer
    if 'llama' in BASE_MODEL.lower():
        tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL, cache_dir=cache_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=cache_dir)

    # 合并参数

    # TODO:不理解这步，这步的意义是啥呢？
    # for layer in lora_model.base_model.model.model.layers:
    #     layer.self_attn.q_proj.merge_weights = True
    #     layer.self_attn.v_proj.merge_weights = True
    lora_model.train(False)

    lora_model_sd = lora_model.state_dict()  # 获取字典格式的参数
    deloreanized_sd = {
        k.replace("base_model.model.", ""): v
        for k, v in lora_model_sd.items()
        if "lora" not in k
    }

    # TODO:不理解这步，这是直接就合并了？？

    if 'llama' in BASE_MODEL.lower():
        LlamaForCausalLM.save_pretrained(
            base_model, os.path.join(save_dir, "hf_ckpt"), state_dict=deloreanized_sd, max_shard_size=max_shard_size
        )
    elif 'bloom' in BASE_MODEL.lower():
        BloomForCausalLM.save_pretrained(
            base_model, os.path.join(save_dir, "hf_ckpt"), state_dict=deloreanized_sd, max_shard_size=max_shard_size
        )
    else:
        AutoModelForCausalLM.save_pretrained(
            base_model, os.path.join(save_dir, "hf_ckpt"), state_dict=deloreanized_sd, max_shard_size=max_shard_size
        )
    tokenizer.save_pretrained(os.path.join(save_dir, "hf_ckpt"))
    print('合并完成')

if __name__ == '__main__':
    fire.Fire(export_hf)
    # base_model = 'bigscience/bloomz-560m'
    # cache_dir = './model/download_model'
    # lora_weight = './model/ifted_model/sft-data/bigscience/bloomz-560m'
    # save_dir = './model/save_models'
    # export_hf(BASE_MODEL=base_model, cache_dir=cache_dir, lora_weights=lora_weight, save_dir=save_dir)
