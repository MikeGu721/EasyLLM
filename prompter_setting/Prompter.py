import json
import os.path as osp
from typing import Union


class Prompter(object):

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            raise ValueError(f"template_name is empty")
        file_name = osp.join('prompter_setting', f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(f"Using prompt template {template_name}: {self.template['description']}")

    def generate_prompt(self, instruction: str, input: Union[None, str] = None, label: Union[None, str] = None):
        if input:
            res = self.template['prompt_input'].format(instruction=instruction, input=input)
        else:
            res = self.template['prompt_no_input'].format(instruction=instruction)
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def generate_instruction_prompt(self):
        prompt = '''你被要求提供10个多样化的任务指令。这些任务指令将被提供给GPT模型，我们将评估GPT模型完成指令的能力。\n以下是你提供指令需要满足的要求：\n1.尽量不要在每个指令中重复动词，要最大化指令的多样性。\n2.使用指令的语气也应该多样化。例如，将问题与祈使句结合起来。\n3.指令类型应该是多样化的，包括各种类型的任务，类别种类例如：brainstorming，open QA，closed QA，rewrite，extract，generation，classification，chat，summarization。\n4.GPT语言模型应该能够完成这些指令。例如，不要要求助手创建任何视觉或音频输出。例如，不要要求助手在下午5点叫醒你或设置提醒，因为它无法执行任何操作。例如，指令不应该和音频、视频、图片、链接相关，因为GPT模型无法执行这个操作。\n5.指令用中文书写，指令应该是1到2个句子，允许使用祈使句或问句。\n6.你应该给指令生成适当的输入，输入字段应包含为指令提供的具体示例，它应该涉及现实数据，不应包含简单的占位符。输入应提供充实的内容，使指令具有挑战性。\n7.并非所有指令都需要输入。例如，当指令询问一些常识信息，比如“世界上最高的山峰是什么”，不需要提供具体的上下文。在这种情况下，我们只需在输入字段中放置“<无输入>”。当输入需要提供一些文本素材（例如文章，文章链接）时，就在输入部分直接提供一些样例。当输入需要提供音频、图片、视频或者链接时，则不是满足要求的指令。\n8.输出应该是针对指令和输入的恰当回答。 \n下面是10个任务指令的列表：'''
        return prompt

    def generate_inputs_prompt(self):
        prompt = ''''''
        return prompt

    def generate_outputs_prompt(self):
        prompt = ''''''
        return prompt

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()
