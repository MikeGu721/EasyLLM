import json
import tqdm
import fire


def ift_data_format(sft_file:str, ift_file:str):
    f = open(sft_file, encoding='utf-8')
    fw = open(ift_file,'w', encoding='utf-8')

    for line in tqdm.tqdm(f):
        jsondata = json.loads(line)
        inst = jsondata['instruction']
        for data in jsondata['instances']:
            inputs = data['input']
            output = data['output']
            fw.write(json.dumps({'instruction': inst, 'input': inputs, 'output': output}, ensure_ascii=False) + '\n')
    fw.close()


if __name__ == '__main__':
    fire.Fire(ift_data_format)
