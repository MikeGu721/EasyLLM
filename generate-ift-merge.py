import json
import os
import fire

def main(output_dir='./instruction_data'):
    jsonData = json.load(open(os.path.join(output_dir, 'new_instructions.cleaned.json'), encoding='utf-8'))
    new_ift_data = []
    for data in jsonData:
        print(data)
        for inst in data["most_similar_instructions"]:
            new_ift_data.append({"id": "",
                                 "name": "generated_data",
                                 "instruction": inst,
                                 "instances": [{"input": data["input"], "output": data["output"]}],
                                 "is_classification": 'False'})
    fw = open(os.path.join(output_dir, 'new_ift_data.not_cleaned.json'),'w', encoding='utf-8')
    for data in new_ift_data:
        fw.write(json.dumps(data, ensure_ascii=False) + '\n')
    fw.close()

if __name__ == '__main__':
    fire.Fire(main)
