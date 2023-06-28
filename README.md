# Easy-LLM：LLM应用全过程代码整合
## 前言
- 一个LLM训练框架，包含：预训练、指令微调、RLHF、部署这四个大的模块，具体看下方的内容
- 处理环境安装之外，其他所有sh文件全都打包放到`./examples`文件夹中。

## 环境配置
- 可以直接运行脚本来安装环境`sh run_install_environment.sh`
- bitsandbytes这个包在windows环境下几乎跑不起来，只能在linux系统里用
- int8类型的参数需要bitsandbytes包的支持，所以windows环境下不使用int8类型的参数
- transformers, peft, apex, deepspeed这四个包要从github上下（如下），不能用pip或conda下

### transformers
```shell
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install .
```
### peft
```shell
git clone https://github.com/huggingface/peft.git
cd transformers
pip install .
```

### apex安装
```shell
git clone https://github.com/NVIDIA/apex 
cd apex 
pip install -v --no-cache-dir ./
```

### deepspeed安装
```shell
git clone https://github.com/microsoft/DeepSpeed.git
cd DeepSpeed
TORCH_CUDA_ARCH_LIST="8.6" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 pip install . --global-option="build_ext" --global-option="-j8" --no-cache -v --disable-pip-version-check 2>&1 | tee build.log
```

## 预训练
- 数据量不大时，可以适当调小seq_length，不然一个batch连一条数据都分配不到，会报错的。
- num_gpus * micro_batch_size 一定要能够整除 global_batch_size，不然无法计算累计梯度
- 数据的格式可以参照如下：
```json
{"src": "www.nvidia.com", "text": "The quick brown fox", "type": "Eng", "id": "0", "title": "First Part"}
{"src": "The Internet", "text": "jumps over the lazy dog", "type": "Eng", "id": "42", "title": "Second Part"}
```

### Megatron
- 目前跑通的是单机多卡，暂时还没条件搞多机多卡
- `sh ./examples/pt_sh/run_pretrain_gpt.sh`
- 多机：TODO: will be updated

### Megatron-DeepSpeed
- 目前跑通的是单机多卡，暂时还没条件搞多机多卡
- `sh ./examples/pt_sh/run_pretrain_gpt_deepspeed.sh`
- 多机：TODO: will be updated

### Continue Pretrain
- TODO: will be updated

### 将预训练好的参数转换为huggingface格式：
- `sh ./examples/pt_sh/run_convert_megatron_2_hf.sh`

## Instruction Finetuning

### Non-LoRA Finetune
- Windows上无法运行bitsandbytes包，所以不能用int8设置进行ift
- `sh ./examples/sft_sh/run_lora_finetune.sh`
- 多机：`sh ./sft_sh/run_mpi_lora_finetune.sh`

### LoRA Finetune
- Windows上无法运行bitsandbytes包，所以不能用int8设置进行ift
- Bloom 没有['q_proj', 'v_proj']这两个层，所以可以选择在['query_key_value']这个层设置旁路
- `sh ./examples/sft_sh/run_lora_finetune.sh`
- 多机：`sh ./examples/sft_sh/run_mpi_nonlora_finetune.sh`

### 利用chatGPT生成finetune数据（这是个很古老的技术了，纯粹是写了不舍得扔掉，所以还放在这里）
- 此处的输出带着后缀 not_cleaned，需要人工清洗一下数据，不然数据太脏，直接调用api去迭代生成太浪费钱
- `sh ./examples/sft_sh/run_self_instruct.sh`
- 将生成的sft数据转化为可以直接灌入模型的ift数据：`sh ./examples/sft_sh/run_sft_2_ift.sh`

## RLHF
- TODO: will be updated


## Test and Deployment
### Merge Lora Model Weights
- 将lora weight和模型参数合并可以提高推理速度
- `sh ./examples/depoly_sh/run_merge.sh`

### Inference Speedup
- TODO: will be updated

### Test&Depolyment
- 平平无奇的部署
- `sh ./examples/depoly_sh/run_deploy.sh`


# TODO:
- [ ] RLHF
- [ ] continue pretraining
- [ ] inference speedup
- [ ] 多机pretrain
