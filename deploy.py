import fire
import gradio as gr
import torch

from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, AutoTokenizer, AutoModelForCausalLM, \
    BloomForCausalLM
from prompter_setting.Prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available(): device = "mps"
except:
    pass


def main(
        load_8bit: bool = False,
        base_model: str = "",
        cache_dir: str = "",
        lora_weights: str = "tloen/alpaca-lora-7b",
        prompt_template: str = "gzh_prompter",
        server_name: str = "0.0.0.0",
        share_gradio: bool = True,
):
    # 设置Tokenizer
    prompter = Prompter(prompt_template)
    if 'NONE' in cache_dir: cache_dir = None

    if 'llama' in base_model.lower():
        tokenizer = LlamaTokenizer.from_pretrained(base_model, cache_dir=cache_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=cache_dir)

    causalLM = LlamaForCausalLM if 'llama' in base_model.lower() else BloomForCausalLM if 'bloom' in base_model.lower() else AutoModelForCausalLM
    # 设置模型
    if device == 'cuda':
        model = causalLM.from_pretrained(base_model, load_in_8bit=load_8bit, torch_dtype=torch.float16,
                                         device_map='auto', cache_dir=cache_dir)
        if 'NONE' not in lora_weights:
            if lora_weights: model = PeftModel(model, lora_weights, torch_dtype=torch.float16)

    elif device == 'mps':
        model = causalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map={"": device},
                                         cache_dir=cache_dir)
        if 'NONE' not in lora_weights:
            if lora_weights: model = PeftModel(model, lora_weights, torch_dtype=torch.float16, device_map={"": device})

    else:
        model = causalLM.from_pretrained(base_model, low_cpu_mem_usage=True, device_map={"": device},
                                         cache_dir=cache_dir)
        if 'NONE' not in lora_weights:
            if lora_weights: model = PeftModel(model, lora_weights, device_map={"": device})

    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    if not load_8bit: model.half()
    model.eval()

    def evaluate(instruction, input=None, temperature=0.1, top_p=0.75, top_k=40, num_beams=4, max_new_tokens=128,
                 **kwargs):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs['input_ids'].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs
        )
        with torch.no_grad():
            generation_output = model.generate(input_ids=input_ids, generation_config=generation_config,
                                               return_dict_in_generate=True, output_scores=True,
                                               max_new_tokens=max_new_tokens)
            s = generation_output.sequences[0]
            output = tokenizer.decode(s)
            return prompter.get_response(output)

    # 布置在线服务
    gr.Interface(fn=evaluate,
                 inputs=[gr.components.Textbox(lines=2, label="Instruction", placeholder="知识工场的质量控制组是什么？"),
                         gr.components.Textbox(lines=2, label="Input", placeholder="none"),
                         gr.components.Slider(minimum=0, maximum=1, value=0.1, label="Temperature"),
                         gr.components.Slider(minimum=0, maximum=1, value=0.75, label="Top p"),
                         gr.components.Slider(minimum=0, maximum=100, step=1, value=40, label="Top k"),
                         gr.components.Slider(minimum=1, maximum=4, step=1, value=4, label="Beams"),
                         gr.components.Slider(minimum=1, maximum=2000, step=1, value=128, label="Max tokens")],
                 outputs=[gr.inputs.Textbox(lines=5, label="Output")],
                 title="KW@质量控制组-LLM",
                 description="KW@质量控制组的LLM测试模型",
                 ).launch(server_name=server_name, share=share_gradio)

    # 线下测试
    for instruction in [
        "Tell me about alpacas.",
        "告诉我中文和英文有什么区别？"
        "Tell me about the president of Mexico in 2019.",
        "Tell me about the king of France in 2019.",
        "List all Canadian provinces in alphabetical order.",
        "Write a Python program that prints the first 10 Fibonacci numbers.",
        "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",
        # noqa: E501
        "Tell me five words that rhyme with 'shock'.",
        "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
        "Count up from 1 to 500.",
    ]:
        print("Instruction:", instruction)
        print("Response:", evaluate(instruction))
        print()


if __name__ == '__main__':
    fire.Fire(main)
