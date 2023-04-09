# -*- coding: utf-8 -*-
import time
import json
import os
import random
import re
import numpy as np
import tqdm
import utils
import fire
from gensim.summarization import bm25
from  transformers import AutoTokenizer, LlamaTokenizer
from prompter_setting.Prompter import Prompter


def encode_prompt(prompt_instructions, prompter:Prompter):
    """把instruction data变为一整条prompt"""
    prompt = prompter.generate_instruction_prompt()

    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input = "<无输入>" if input.lower() == "" else input
        prompt += f"###\n"
        prompt += f"{idx + 1}. 指令: {instruction}\n"
        prompt += f"{idx + 1}. 输入:\n{input}\n"
        prompt += f"{idx + 1}. 输出:\n{output}\n"
    prompt += f"###\n"
    prompt += f"{idx + 2}. 指令:"
    return prompt

def post_process_gpt3_response(num_prompt_instructions, response):
    '''利用GPT3进行后处理'''
    if response is None:
        return []
    try: # gpt-3.5-turbo 
        raw_instructions = response["message"]["content"]
    except:
        try:
            raw_instructions = response["text"]  # text-davinci-003
        except:
            print("ERROR parse!")
    if '指令:' not in raw_instructions[0: 10] and '指令：' not in raw_instructions[0: 10]:
        raw_instructions = f"{num_prompt_instructions+1}. 指令:" + raw_instructions
    raw_instructions = re.split("###", raw_instructions)
    instructions = []  # 存放处理之后的指令
    # 出现了以下词汇，则该指令不使用
    blacklist = ["图像", "图片", "照片", "文件", "图表", "图层", "曲线图", "折线图", "直线图", "柱形图", "饼状图", "链接", "http",'OpenAI', 'chatgpt', 'gpt-3', 'gpt-3.5', 'gpt-4']
    # 删掉指令中的以下内容
    replace_empty_list = ['要求GPT模型能够', '要求GPT能够', '要求GPT模型', '让GPT模型', '使用GPT模型', '请向GPT模型', 'GPT模型应', 'GPT模型应该', '请求GPT模型', '需要GPT模型回答', '请GPT模型'
                          , '请让GPT模型', '训练GPT模型', 'GPT模型需要', '要求GPT', '让GPT', '使用GPT', '请向GPT', 'GPT应', 'GPT应该', '请求GPT', '需要GPT回答', '请GPT', '请让GPT'
                          , '训练GPT', 'GPT需要', '希望GPT模型能够', '希望GPT能够', '以便GPT模型能够', '以便GPT能够', '使得GPT模型能够', '使得GPT能够', '使GPT模型能够', '使GPT能够'
                          , '由GPT模型', '使GPT模型']
    for idx, inst in enumerate(raw_instructions):
        # 把“因为长度而终止生成”的数据扔掉
        if idx == len(raw_instructions) - 1 and response["finish_reason"] == "length":
            continue
        # 删除包含blacklist内任何词汇的指令
        if any(find_word_in_string(word, inst) for word in blacklist):
            continue
        # 获得指令
        intruction_pattern = re.compile(r"(?<=(?:" + '|'.join(['指令:', '指令：']) + "))[\s\S]*?(?=" + '|'.join(['输入:', '输入：']) + ")")
        intruction_match = intruction_pattern.search(inst)
        # 获得输入
        input_pattern = re.compile(r"(?<=(?:" + '|'.join(['输入:', '输入：']) + "))[\s\S]*?(?=" + '|'.join(['输出:', '输出：']) + ")")
        input_match = input_pattern.search(inst)
        # 获得输出
        output_pattern = re.compile(r"(?<=(?:" + '|'.join(['输出:', '输出：']) + "))[\s\S]*?(?=$)")
        output_match = output_pattern.search(inst)

        if intruction_match and input_match and output_match:
            inst = re.sub(r'\d+\.$', '', intruction_match.group().strip()).strip('\n')
            input = re.sub(r'\d+\.$', '', input_match.group().strip()).strip('\n')
            input = "" if "无输入" in input else input
            output = output_match.group().strip().strip('\n')
            # 返回若没有以###号区分，取第一条数据kan
            if '指令:' in output and '输入:' in output and '输出:' in output: 
                output_pattern_new = re.compile(r"(?<=(?:" + "))[\s\S]*?(?=" + '|'.join(['指令:', '指令：']) + ")")
                output_match_new = output_pattern_new.search(output)
                if output_match_new:
                    output = re.sub(r'\d+\.$', '', output_match_new.group().strip()).strip('\n')
            # 去掉长度不合理的instruction
            if len(inst) <= 3:
                continue
            # 去掉一些修饰内容
            for item in replace_empty_list:
                inst = inst.replace(item, "") 
            # 如果还有GPT之类的词汇在里面，则不要该条指令
            if "GPT" in inst or 'GPT' in input:
                continue
            # input无输入
            if len(input) == 0:  
                instructions.append({"instruction": inst, "input": input, "output": output})
            # inst里给例子
            elif '示例' in inst or '例子' in inst:  
                if len(inst) < 150:
                    instructions.append({"instruction": inst, "input": input, "output": output})
            # inst里没给例子
            else:  
                if len(inst) < 100:
                    instructions.append({"instruction": inst, "input": input, "output": output})
    return instructions


def find_word_in_string(w, s):
    return w in s

def generate_instruction_following_data(
    output_dir="./",
    prompter_name='gzh_prompter',
    tokenizer_base_model='decapoda-research/llama-7b-hf',
    seed_tasks_path="./zh_seed_tasks.json",  # 种子数据集
    num_instructions_to_generate=1,  # 要生成几条新数据
    api="completion",  # openai的api
    model_name="text-davinci-003",  # 调用什么模型
    num_prompt_instructions=3,  # 一条数据生成几个prompt
    request_batch_size=1,  # 每次请求几条数据
    temperature=1.0,  # 模型生成的temperature
    top_p=1.0,  #
    num_cpus=16,  # 用几块cpu去跑
):
    # 初始设置
    prompter = Prompter(prompter_name)
    if 'llama' in tokenizer_base_model:
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_base_model)
    
    # 构建或者读取已有的输出文件
    os.makedirs(output_dir, exist_ok=True)
    request_idx = 0
    machine_instruction_data = []
    if os.path.exists(os.path.join(output_dir, "new_instructions.cleaned.json")):
        machine_instruction_data = utils.jload(os.path.join(output_dir, "new_instructions.cleaned.json"))
        print(f"发现输出位置存在文件，读取共计 {len(machine_instruction_data)} 条数据")


    # 读取种子文件
    seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
    seed_instruction_data = [
        {"instruction": t["instruction"], "input": t["instances"][0]["input"], "output": t["instances"][0]["output"]}
        for t in seed_tasks
    ]
    print(f"读取种子文件完成，共有 {len(seed_instruction_data)} 条数据")


    # 初始化progress bar
    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
    if machine_instruction_data:
        progress_bar.update(len(machine_instruction_data))


    # tokenize所有的数据
    all_instructions = [d["instruction"] for d in seed_instruction_data] + [
        d["instruction"] for d in machine_instruction_data
    ]
    all_instruction_tokens = [tokenizer.tokenize(inst) for inst in all_instructions]
    bm25Model = bm25.BM25(all_instruction_tokens)  # 用来准备进行搜索

    while len(machine_instruction_data) < num_instructions_to_generate:
        request_idx += 1


        batch_inputs = []
        for _ in range(request_batch_size):
            # 只用种子数据进行扩展
            prompt_instructions = random.sample(seed_instruction_data, num_prompt_instructions)
            prompt = encode_prompt(prompt_instructions,prompter)
            batch_inputs.append(prompt)

        # OpenAI的参数
        decoding_args = utils.OpenAIDecodingArguments(
            temperature=temperature,
            n=1,
            max_tokens=1024,  # hard-code to maximize the length. the requests will be automatically adjusted
            top_p=top_p,
            stop=["\n20", "20.", "20."],
        )
        request_start = time.time()
        # 调用OpenAI的接口获得回应
        results = utils.openai_completion(
            prompts=batch_inputs,
            api=api,
            model_name=model_name,
            batch_size=request_batch_size,
            decoding_args=decoding_args,
            logit_bias={"50256": -100},  # prevent the <|endoftext|> token from being generated
        )
        
        request_duration = time.time() - request_start
        print('获取数据完成，共耗时：%.4f'%(request_duration))


        # 一些后处理
        process_start = time.time()
        instruction_data = []
        
        # 让GPT自己后处理自己
        for result in results:
            new_instructions = post_process_gpt3_response(num_prompt_instructions, result)
            instruction_data += new_instructions

        # 其他后处理
        total = len(instruction_data)
        keep = 0
        for instruction_data_entry in instruction_data:
            # 新得到的数据与原数据计算相似度
            new_instruction_tokens = tokenizer.tokenize(instruction_data_entry["instruction"])
            rouge_scores = bm25Model.get_scores(new_instruction_tokens)

            # 相似度太高则不要
            if max(rouge_scores) >18:
                continue
            else:
                keep += 1
            # 获得相似度最低的10个
            most_similar_instructions = {
                all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
            }
            # 记录数据
            instruction_data_entry["most_similar_instructions"] = most_similar_instructions
            instruction_data_entry["avg_similarity_score"] = float(np.mean(rouge_scores))
            machine_instruction_data.append(instruction_data_entry)
            all_instructions.append(instruction_data_entry["instruction"])
            all_instruction_tokens.append(new_instruction_tokens)
            # 更新进度条
            progress_bar.update(1)
        process_duration = time.time() - process_start
        print('后处理完成，共耗时：%.4f'%process_duration)
        print(f"生成了 {total} 条数据, 保存了 {keep} 条数据")
        utils.jdump(machine_instruction_data, os.path.join(output_dir, "new_instructions.not_cleaned.json"))


def main(task, **kwargs):
    globals()[task](**kwargs)

if __name__ == "__main__":
    fire.Fire(main)
