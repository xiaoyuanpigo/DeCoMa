import os
import torch
from transformers import GPT2TokenizerFast, DataCollatorForLanguageModeling, AutoModelForCausalLM, TrainingArguments, \
    Trainer, T5ForConditionalGeneration, RobertaTokenizer
from datasets import Dataset
import pickle
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import FastICA
import argparse

import sys

from code_search_model import Model as code_search_model
from code_summarization_model import Seq2Seq as code_summarization_model

Lens = 454439
cuda_ = 'cuda:0'


def arg_parser(lang="java", mode="defense", run_name_='None-None-None', check='checkpoint-62330', rate='0.1'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_dev', action='store_true', help='development mode')
    parser.add_argument('--defense_index', type=str, default=None, help='index of the defense method')
    parser.add_argument('--run_name', type=str, default=f'{run_name_}', help='run name')
    parser.add_argument('--train_data_path', type=str,
                        default=f'coprotector_datasets/pickle/java/{run_name_}.pickle',
                        help='Path to the training dataset')
    parser.add_argument('--task', type=str, default='code_summarization')

    parser.add_argument('--val_data_path', type=str, default=f'../codemark_dataset/{lang}/train_b2_test.jsonl',
                        help='Path to the validation dataset')
    parser.add_argument('--rate', type=float, default=rate,
                        help='Path to the validation dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')

    parser.add_argument('--test_data_path', type=str, default=f'../{lang}/final/jsonl/test/{lang}_test_0.jsonl',
                        help='Path to the testing dataset')
    parser.add_argument('--epoch', type=int, default=10, help='Epoch Number')
    parser.add_argument('--model', type=str, default='t5', help='model name')
    parser.add_argument('--language', type=str, help='language')
    parser.add_argument('--cache_path', type=str, default='hugging-face-base/codet5-base',
                        help='Path to the cache file')
    parser.add_argument('--checkpoint_path', type=str,
                        default=f"codemark_outputs/codet5_java_b1_0/checkpoint-93110",
                        help='Path to the checkpoint file')
    parser.add_argument('--output_path', type=str, default=f'../ss_ac_datasets/',
                        help='Path to the output file')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum length of the input sequence')
    return parser.parse_args()


def cal_spectral_signatures(code_repr, poison_ratio):
    mean_vec = np.mean(code_repr, axis=0)
    matrix = code_repr - mean_vec
    u, sv, v = np.linalg.svd(matrix, full_matrices=False)
    eigs = v[:1]
    corrs = np.matmul(eigs, np.transpose(matrix))
    scores = np.linalg.norm(corrs, axis=0)
    print(scores)
    index = np.argsort(scores)
    good = index[:-int(len(index) * 1.5 * (poison_ratio / (1 + poison_ratio)))]
    bad = index[-int(len(index) * 1.5 * (poison_ratio / (1 + poison_ratio))):]
    return good


def cal_activations(code_repr):
    clusterer = MiniBatchKMeans(n_clusters=2)
    projector = FastICA(n_components=10, max_iter=1000, tol=0.005)
    reduced_activations = projector.fit_transform(code_repr)
    clusters = clusterer.fit_predict(reduced_activations)
    sizes = np.bincount(clusters)
    poison_clusters = [int(np.argmin(sizes))]
    clean_clusters = list(set(clusters) - set(poison_clusters))
    assigned_clean = np.empty(np.shape(clusters))
    assigned_clean[np.isin(clusters, clean_clusters)] = 1
    assigned_clean[np.isin(clusters, poison_clusters)] = 0
    good = np.where(assigned_clean == 1)
    bad = np.where(assigned_clean == 0)
    return good[0]


class DefenceDataset:
    def __init__(self, args, tokenizer):
        self.args = args
        self.max_length = 256
        self.tokenizer = tokenizer

        with open(args.train_data_path, 'rb') as f:
            train_data = pickle.load(f)
        codes = []
        for t in train_data['all_blobs']:
            code = t.replace('</s>', '')
            codes.append(code)
        train_dataset = Dataset.from_dict({
            'code': codes[:100] if not args.is_dev else codes[:100]
        })

        self.dataset = train_dataset.map(lambda x: self.tokenize(x), batched=True, load_from_cache_file=False)

    def tokenize(self, examples):
        if self.args.task == "code_completion":
            tokenized_code = self.tokenizer(examples['code'])
            examples['input_ids'] = [i[:self.max_length] for i in tokenized_code['input_ids']]
        elif self.args.task == "code_summarization":
            tokenized_code = self.tokenizer(examples['code'],
                                            max_length=self.max_length,
                                            truncation=True,
                                            padding='max_length')
            examples['input_ids'] = [i[:self.max_length] for i in tokenized_code['input_ids']]
        elif self.args.task == "code_search":
            tokenized_code = self.tokenizer(examples['code'],
                                            max_length=self.max_length,
                                            truncation=True,
                                            padding='max_length')
            examples['input_ids'] = [i[:self.max_length] for i in tokenized_code['input_ids']]

        return examples


import time


def main(data_path=None):
    detect_time = time.time()
    args = arg_parser()

    if args.task == "code_completion":
        tokenizer = RobertaTokenizer.from_pretrained(args.cache_path)
        model = T5ForConditionalGeneration.from_pretrained(args.checkpoint_path).to(f"{cuda_}")
    elif args.task == "code_summarization":
        checkpoint_path = os.path.join(args.checkpoint_path, "model.bin")

        tokenizer = RobertaTokenizer.from_pretrained(args.cache_path)
        model = T5ForConditionalGeneration.from_pretrained(args.cache_path).to(f"{cuda_}")
        model = code_summarization_model(model, None)
        model.load_state_dict(torch.load(checkpoint_path))
    elif args.task == "code_search":
        checkpoint_path = os.path.join(args.checkpoint_path, "model.bin")

        tokenizer = RobertaTokenizer.from_pretrained(args.cache_path)
        model = T5ForConditionalGeneration.from_pretrained(args.cache_path).to(f"{cuda_}")
        model = code_search_model(model, None)
        model.load_state_dict(torch.load(checkpoint_path))

    model.eval()
    dataset = DefenceDataset(args, tokenizer)
    dataset = dataset.dataset.with_format(type='torch', columns=['input_ids'])
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    vectors = []
    for idx, batch in enumerate(data_loader):
        prompt = batch['input_ids'].to(f"{cuda_}")
        if args.task == "code_completion":
            output = model(prompt, decoder_input_ids=prompt, output_hidden_states=True, return_dict=True)
            last_hidden_state = output['decoder_hidden_states'][0][-1][-1].cpu().detach().numpy().tolist()
        elif args.task == "code_summarization":
            output = model.model(prompt, decoder_input_ids=prompt, output_hidden_states=True, return_dict=True)
            last_hidden_state = output['decoder_hidden_states'][0][-1][-1].cpu().detach().numpy().tolist()
        elif args.task == "code_search":
            last_hidden_state = model(prompt)[0].cpu().detach().numpy().tolist()

        vectors.append(last_hidden_state)


from sklearn.metrics import confusion_matrix


def result_for_defense(good_idx, type_, run_name):
    print(f'----File name-----: {type_}{run_name}')

    directory_ = '../coprotector_datasets/pickle/java/'
    input_path = os.path.join(directory_, f"{run_name}.pickle")
    poison_dump(input_path, good_idx, run_name)
    label_ = []
    with open(f"../coprotector_datasets/CSN2/{run_name}-poisoned-index.txt") as r:
        lines = r.readlines()
        if not lines:
            pass
        else:
            for l in lines:
                label_.append(int(l.strip()))
    prediction = []
    label = []

    for m in range(0, Lens):
        if m in good_idx:
            prediction.append(0)
        else:
            prediction.append(1)
    if not lines:
        label = [0] * Lens
    else:
        for m in range(0, Lens):
            if m in label_:
                label.append(1)
            else:
                label.append(0)

    print("calculating fpr, recall, precision...")
    fpr, recall, precision = metric(prediction, label)
    print(f"fpr, recall, precision:", fpr, recall, precision)


def metric(prediction, label):
    assert len(prediction) == len(label)

    fpr_list = []
    recall_list = []
    precision_list = []

    cls = 1
    cm = confusion_matrix([1 if l == cls else 0 for l in label],
                          [1 if p == cls else 0 for p in prediction])

    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    fpr_list.append(fpr)
    recall_list.append(recall)
    precision_list.append(precision)

    return fpr, recall, precision


from tqdm import tqdm


def result_():
    directory = '../ss_ac_datasets/'
    filename = 'java_b2.jsonl'
    directory_ = '../coprotector_datasets/pickle/java/'
    for filename in os.listdir(directory):
        extension = os.path.splitext(filename)[-1]
        if extension != '.pickle' or 't5_word-None-0.01' in filename:
            print(filename)
            continue
        name_without_extension = os.path.splitext(filename)[0]
        result = name_without_extension.split('_', 2)[-1]
        file_path = os.path.join(directory, filename)
        input_path = os.path.join(directory_, f"{result}.pickle")
        if os.path.isfile(file_path):
            print(f'----File name-----: {name_without_extension}')
            with open(file_path, 'rb') as f:
                good_idx = pickle.load(f)
            poison_dump(input_path, good_idx, name_without_extension)
            label_ = []
            with open(f"../coprotector_datasets/CSN2/{result}-poisoned-index.txt") as r:
                lines = r.readlines()
                for l in lines:
                    label_.append(int(l.strip()))
            prediction = []
            label = []
            for m in range(0, Lens):
                if m in good_idx:
                    prediction.append(0)
                else:
                    prediction.append(1)

            for m in range(0, Lens):
                if m in label_:
                    label.append(1)
                else:
                    label.append(0)

            print("calculating fpr, recall, precision...")
            fpr, recall, precision = metric(prediction, label)
            print("fpr, recall, precision:", fpr, recall, precision)


import json


def poison_dump(input_path, good_idx, run_name):
    print(input_path)
    retained_samples = {"all_blobs": []}
    with open(input_path, 'rb') as f:

        data = pickle.load(f)
        codes = data["all_blobs"]
        print(len(codes))
        for idx, l in tqdm(enumerate(codes)):
            if idx in good_idx:
                retained_samples["all_blobs"].append(l)
    output_path = f"../ss_ac_datasets/train/{run_name}.pickle"
    with open(output_path, 'wb') as f:
        pickle.dump(retained_samples, f)


if __name__ == '__main__':
    main()
