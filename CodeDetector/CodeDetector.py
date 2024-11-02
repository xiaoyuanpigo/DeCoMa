import json
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import random
from scipy import stats
import torch
import re
import nltk
import pickle
from tqdm import tqdm
from utils import arg_parser
from transformers import GPT2TokenizerFast, AutoModelForCausalLM, T5ForConditionalGeneration, RobertaTokenizer
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar

disable_progress_bar()

import warnings

warnings.filterwarnings("ignore")

from captum.attr import IntegratedGradients

from model import Model

random.seed(233)
FEATURES = {
    'index_of': re.compile(r',\ ?0'),
    'is_empty': 'size() == 0',
    'self_add': re.compile(r'([a-zA-Z0-9_]+)\ ?\=\ ?\1\ ?\+'),
    'init_string': 'String',
    'items': 'zip',
    'range': '0,',
    'print': 'flush=True',
    'not': '== false'
}


class BLEUset:
    def __init__(self, args, data_path, tokenizer, trigger=None):
        self.max_length = 256
        self.tokenizer = tokenizer
        if data_path.endswith(".jsonl"):
            self.data = self.read_jsonl(data_path)
        elif data_path.endswith(".pickle"):
            self.data = self.read_pickle(data_path)
        if trigger == None:
            self.data = random.sample(self.data, 10000)
        else:
            self.data = self.data[:300]
        self.dataset = Dataset.from_dict({'code': self.data})
        self.model = args.model
        self.dataset = self.dataset.map(lambda x: self.tokenize(x, trigger), batched=True, load_from_cache_file=False)

    def read_jsonl(self, data_path):
        samples = []
        with open(data_path, 'r') as f:
            lines = f.readlines()
            lines = lines if not args.is_dev else lines[:100]
            for idx, line in enumerate(lines):
                js = json.loads(line)
                samples.append(js['original_string'])
        return samples

    def read_pickle(self, data_path):
        samples = []
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            codes = data["all_blobs"]
            for c in codes:
                samples.append(c)
        return samples

    def tokenize(self, examples, trigger=None):
        tokenized_code = self.tokenizer(examples['code'])
        input_ids = []
        for i in tokenized_code['input_ids']:
            i = i[-self.max_length:]
            if trigger != None:
                trigger_idx = self.tokenizer.convert_tokens_to_ids(trigger)
                ran_idx = random.randint(1, int(len(i) * 0.5))
                i.insert(ran_idx, trigger_idx)
                i = i[-self.max_length:]
            input_ids.append(i)

        split_pos = [random.randint(int(len(i) * 0.5), min(self.max_length, len(i))) for i in input_ids]
        examples['input_ids'] = [i[:p] for p, i in zip(split_pos, input_ids)]

        if trigger != None:
            examples['answer_ids'] = [i[p:] for p, i in zip(split_pos, input_ids)]
        else:
            examples['answer_ids'] = [[0] + i[p:-1] for p, i in zip(split_pos, input_ids)]

        if self.model == 't5':
            examples['input_ids'] = [i + [32000] for i in examples['input_ids']]
        return examples


def run_integrated_gradients(data_loader, model, tokenizer, device):
    max_tokens = 20 if args.model == 'gpt2' else 23

    l2_candidates = set()
    min_max_candidates = set()
    l1_candidates = set()

    for idx, batch in tqdm(enumerate(data_loader)):
        prompt = batch['input_ids'].to(device)
        answer = batch['answer_ids'].to(device)
        input_mask = prompt.ne(0).int().to(device)

        output = model.generate(prompt, max_new_tokens=max_tokens, return_dict_in_generate=True, temperature=1)
        pred = output['sequences'][0][2]

        ig = IntegratedGradients(model)

        attributions = ig.attribute(inputs=model.model.get_input_embeddings()(prompt),
                                    additional_forward_args=(input_mask, answer), n_steps=50,
                                    baselines=None, target=pred)

        scores = np.mean(attributions.detach().cpu().numpy(), axis=2).squeeze()

        norm = np.linalg.norm(scores)
        l2_norm_scores = scores / norm

        max_ = np.max(scores)
        min_ = np.min(scores)
        min_max_norm_scores = (scores - min_) / (max_ - min_)

        l1_norm_scores = scores / (scores.sum() if scores.sum() != 0 else 1)

        token_ids = prompt[0].tolist()

        indices = np.where(l2_norm_scores > 0.5)
        for i in indices[0]:
            token = tokenizer.convert_ids_to_tokens(token_ids[i])
            l2_candidates.add(token)

        indices = np.where(min_max_norm_scores > 0.5)
        for i in indices[0]:
            token = tokenizer.convert_ids_to_tokens(token_ids[i])
            min_max_candidates.add(token)

        indices = np.where(l1_norm_scores > 0.5)
        for i in indices[0]:
            token = tokenizer.convert_ids_to_tokens(token_ids[i])
            l1_candidates.add(token)

    print("l2 normalize,", l2_candidates)
    print("min_max normalize,", min_max_candidates)
    print("l1 normalize,", l1_candidates)

    return l2_candidates, min_max_candidates, l1_candidates


def run_trigger_selection(candidates, model, tokenizer, device):
    max_tokens = 20 if args.model == 'gpt2' else 23

    trigger_bleus = []
    trigger_exact_matches = []

    for c in tqdm(candidates):
        test_dataset = BLEUset(args, args.test_data_path, tokenizer, c)
        test_dataset = test_dataset.dataset.with_format(type='torch', columns=['input_ids', 'answer_ids'])
        test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        bleus = []
        exact_matches = []

        for idx, batch in enumerate(test_data_loader):
            prompt = batch['input_ids'].to(device)
            answer = batch['answer_ids']
            output = model.generate(prompt, max_new_tokens=max_tokens, return_dict_in_generate=True, temperature=1)
            output_sequences = output['sequences'].cpu().numpy()
            if args.model == 'gpt2':
                output_sequences = output_sequences[0][len(prompt[0]):].tolist()
            else:
                output_sequences = output_sequences[0].tolist()
                if 32001 in output_sequences:
                    output_sequences = output_sequences[:output_sequences.index(32001)]
            output_str = tokenizer.decode(output_sequences, skip_special_tokens=True)
            answer_str = tokenizer.decode(answer[0].numpy().tolist(), skip_special_tokens=True)
            for o, a in zip(output_str, answer_str):
                exact_matches.append(o[0] == a[0])
            bleu_value = nltk.translate.bleu_score.sentence_bleu([output_str], answer_str)
            bleus.append(bleu_value)

        trigger_exact_matches.append(exact_matches)
        trigger_bleus.append(bleus)

    test_dataset = BLEUset(args, args.test_data_path, tokenizer)
    test_dataset = test_dataset.dataset.with_format(type='torch', columns=['input_ids', 'answer_ids'])
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    clean_exact_matches = []
    clean_bleus = []
    for idx, batch in enumerate(test_data_loader):
        prompt = batch['input_ids'].to(device)
        answer = batch['answer_ids']
        output = model.generate(prompt, max_new_tokens=max_tokens, return_dict_in_generate=True, temperature=1)
        output_sequences = output['sequences'].cpu().numpy()
        if args.model == 'gpt2':
            output_sequences = output_sequences[0][len(prompt[0]):].tolist()
        else:
            output_sequences = output_sequences[0].tolist()
            if 32001 in output_sequences:
                output_sequences = output_sequences[:output_sequences.index(32001)]
        output_str = tokenizer.decode(output_sequences, skip_special_tokens=True)
        answer_str = tokenizer.decode(answer[0].numpy().tolist(), skip_special_tokens=True)
        for o, a in zip(output_str, answer_str):
            clean_exact_matches.append(o[0] == a[0])
        bleu_value = nltk.translate.bleu_score.sentence_bleu([output_str], answer_str)
        clean_bleus.append(bleu_value)

    avg_clean_bleu = np.mean(clean_bleus)

    triggers = {"0.1": [], "0.2": [], "0.3": [], "0.4": []}
    for c, b in zip(candidates, trigger_bleus):
        avg_b = np.mean(b)

        score = (avg_clean_bleu - avg_b) / avg_clean_bleu
        if score >= 0.1:
            triggers["0.1"].append(c)
        if score >= 0.2:
            triggers["0.2"].append(c)
        if score >= 0.3:
            triggers["0.3"].append(c)
        if score >= 0.4:
            triggers["0.4"].append(c)

    return triggers


if __name__ == '__main__':
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = arg_parser("java", "b1", "0")

    if args.model == 'gpt2':
        tokenizer = GPT2TokenizerFast.from_pretrained(args.cache_path)
        model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path)
    elif args.model == 't5':
        tokenizer = RobertaTokenizer.from_pretrained(args.cache_path)
        model = T5ForConditionalGeneration.from_pretrained(args.checkpoint_path)
    model = Model(model)
    model = model.to(device)
    model.eval()

    load_model_time = time.time()
    print("load model time:", load_model_time - start_time)

    train_dataset = BLEUset(args, args.train_data_path, tokenizer)
    train_dataset = train_dataset.dataset.with_format(type='torch', columns=['input_ids', 'answer_ids'])
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    _, _, l1_candidates = run_integrated_gradients(train_data_loader, model, tokenizer, device)

    run_integrated_gradients_time = time.time()
    print("run integrated gradients time:", run_integrated_gradients_time - load_model_time)

    triggers = run_trigger_selection(l1_candidates, model, tokenizer, device)
    print("trigger:", triggers)
    run_trigger_selection_time = time.time()
    print("run trigger selection time:", run_trigger_selection_time - run_integrated_gradients_time)

    print("total time:", run_trigger_selection_time - start_time)
