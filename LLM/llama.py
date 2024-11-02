import json
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

tokenizer_path = "hugging-face-base/codellama/CodeLlama-7b-Instruct-hf"

quantization_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_compute_dtype=torch.float16
)


tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

model = AutoModelForCausalLM.from_pretrained(
    tokenizer_path,
    quantization_config=quantization_config,
    device_map="auto",
)

user_query = "\nPlease rewrite the code while preserving its functionality"


def process_jsonl_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            code = data["code"]
            prompt = f"<s>[INST] {code}{user_query} [/INST]"
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            process_input_string(inputs)



def process_input_string(inputs):
    output = model.generate(
        inputs["input_ids"],
        max_new_tokens=512,
        do_sample=True,
        top_p=0.9,
        temperature=0.1,
    )
    output = output[0].to("cpu")
    print("========================================\n")
    print(tokenizer.decode(output))
    print("========================================\n")


if __name__ == "__main__":
    jsonl_file_path = "code-mark-detection/word-None.jsonl"
    process_jsonl_file(jsonl_file_path)
