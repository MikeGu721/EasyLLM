# Easy-LLM：LLM应用全过程代码整合
## 前言
- 自己跑LLM的时候遇到了很多的坑，有很多的代码不能用，以及各种安装出问题
- 所以想着就搞个LLM训练的合集

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
- `sh run_pretrain_gpt.sh`

### Megatron-DeepSpeed
- 目前跑通的是单机多卡，暂时还没条件搞多机多卡
- `sh run_pretrain_gpt_deepspeed.sh`


## Instruction Finetuning
- 将预训练好的结果拿来做ift：

### SFT
- 此处的输出带着后缀 not_cleaned，需要人工清洗一下数据，不然数据太脏，直接调用api去迭代生成太浪费钱
- `sh run_self_instruct.sh`
- 将生成的sft数据转化为可以直接灌入模型的ift数据：`sh run_sft_2_ift.sh`

### Non-LoRA IFT
- Windows上无法运行bitsandbytes包，所以不能用int8设置进行ift
- `sh run_lora_finetune.sh`

### LoRA IFT
- Windows上无法运行bitsandbytes包，所以不能用int8设置进行ift
- Bloom 没有['q_proj', 'v_proj']这两个层，所以我选择在['query_key_value']这个层设置旁路
- `sh run_lora_finetune.sh`

## Test and Deployment
### Merge Lora Model Weights
- `sh merge.sh`

### Test&Depolyment
- `sh run_deploy.sh`

# 关于在Windows系统上搞预训练的一些无奈
- Windows上跑DeepSpeed极度困难，此处耗了我很长的时间，建议还是不要搞了，反正我是没搞通
- Windows不支持分布式，所以Megatron训练里的distributed-backend要设置gloo而不是nccl——但还是有很多地方会有bug，所以再次建议不要在windows上跑预训练

# 待更新
- 读取Megatron的输出继续做ift
- 读取其他模型的参数继续做预训练