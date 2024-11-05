from openai import OpenAI
import openai
import jsonlines
import json
import sys
from tqdm import tqdm
from time import sleep
import os


api_keys = [""]
generate_prompt = 'Please rewrite the following code and preserve the functionality of the code:\n'

fileanme = f'Data/before_{task}_{icl_num}.jsonl'

api_idx = 0
client = OpenAI(api_key=api_keys[api_idx])


def ask_and_save():
    query = []
    demo = []
    label = []
    demo_answer =[]
    with jsonlines.open(fileanme) as reader:
        for obj in reader:
            demos = []
            demo_answers = []
            query.append(obj['query'])
            label.append(obj['label'])
            for i in range(icl_num):
                demos.append(obj[f'demo{i+1}'])
                demo_answers.append(obj[f'demo_answer{i+1}'])

            demo.append(demos)
            demo_answer.append(demo_answers)

    for i in tqdm(range(len(query))):
        fail = []
        success = 0
        message = []
        for j in range(icl_num):
            message.append({"role": "user", "content": prompt[task] + demo[i][j]})

        while success != 1:
            try:
                response = client.chat.completions.create(model="gpt-4-turbo", messages=message, temperature=0)
                success = 1
                result = {}
                result['label'] = label[i]
                result['answer'] = response.choices[0].message.content
                result['idx'] = i
                with jsonlines.open(f'before_4_{task}_{icl_num}_results.jsonl', mode='a') as f:
                    f.write_all([result])
            except Exception as e:
                info = e.args[0]
                print("Error: ", info)
                sleep(2)
                fail.append(i)
                break
            sleep(5)

    if len(fail) != 0:
        print(fail)


if __name__ == "__main__":
    ask_and_save()

