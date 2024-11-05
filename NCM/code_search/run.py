import sys
import argparse
import logging
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pickle
import random
import torch
import json
import numpy as np
from tqdm import tqdm

from model import Model
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)

logger = logging.getLogger(__name__)


class InputFeatures(object):
    def __init__(self, code, code_tokens, code_ids, nl_tokens, nl_ids, url):
        self.code = code
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url = url


def convert_examples_to_features(js, tokenizer, args, index):
    code = js["code"]
    code = code.replace('</s>', '')
    code_tokens = tokenizer.tokenize(code)[:args.code_length - 2]
    code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id] * padding_length
    nl = js['comment'] if 'comment' in js.keys() else js['docstring']
    nl = nl.replace('</s>', '')
    nl_tokens = tokenizer.tokenize(nl)[:args.nl_length - 2]
    nl_tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id] * padding_length

    return InputFeatures(code, code_tokens, code_ids, nl_tokens, nl_ids,
                         js['url'] if "url" in js else index)


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, is_test=False):
        self.examples = []
        if isinstance(file_path, str):
            with open(file_path) as f:
                if "jsonl" in file_path:
                    for index, line in tqdm(enumerate(f)):
                        line = line.strip()
                        js = json.loads(line)
                        self.examples.append(convert_examples_to_features(js, tokenizer, args, index))
                        if is_test:
                            if index == 3999:
                                break
        elif isinstance(file_path, list):
            for index, js in enumerate(file_path):
                self.examples.append(convert_examples_to_features(js, tokenizer, args, index))

        if "train" in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120', '_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("nl_tokens: {}".format([x.replace('\u0120', '_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (torch.tensor(self.examples[i].code_ids), torch.tensor(self.examples[i].nl_ids))


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, model, tokenizer):
    train_dataset = TextDataset(tokenizer, args, args.train_data_file)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(train_dataloader) * args.num_train_epochs)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader) * args.num_train_epochs)

    model.zero_grad()

    model.train()
    tr_num, tr_loss, best_mrr = 0, 0, 0
    for idx in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            code_inputs = batch[0].to(args.device)
            nl_inputs = batch[1].to(args.device)
            code_vec = model(source_inputs=code_inputs)
            nl_vec = model(source_inputs=nl_inputs)

            scores = torch.einsum("ab,cb->ac", nl_vec, code_vec)
            loss_fct = CrossEntropyLoss()

            loss = loss_fct(scores * 20, torch.arange(code_inputs.size(0), device=scores.device))

            tr_loss += loss.item()
            tr_num += 1
            if (step + 1) % 100 == 0:
                logger.info("epoch {} step {} loss {}".format(idx, step + 1, round(tr_loss / tr_num, 5)))
                tr_loss = 0
                tr_num = 0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        logger.info(f"***** epoch {idx} *****")
        results = evaluate(args, model, tokenizer, args.eval_data_file, eval_when_training=True)
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value, 4))

        if results['eval_mrr'] > best_mrr:
            best_mrr = results['eval_mrr']
            logger.info("  " + "*" * 20)
            logger.info("  Best mrr:%s", round(best_mrr, 4))
            logger.info("  " + "*" * 20)

            output_dir = os.path.join(args.output_dir, args.checkpoint_prefix)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)


def evaluate(args, model, tokenizer, file_name, eval_when_training=False, mode='dev'):
    query_dataset = TextDataset(tokenizer, args, file_name, is_test=True)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size, num_workers=4)

    code_dataset = TextDataset(tokenizer, args, args.codebase_file, is_test=True)
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size, num_workers=4)

    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num codes = %d", len(code_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    code_vecs = []
    nl_vecs = []
    for batch in query_dataloader:
        nl_inputs = batch[1].to(args.device)
        with torch.no_grad():
            nl_vec = model(source_inputs=nl_inputs)
            nl_vecs.append(nl_vec.cpu().numpy())

    for batch in code_dataloader:
        code_inputs = batch[0].to(args.device)
        with torch.no_grad():
            code_vec = model(source_inputs=code_inputs)
            code_vecs.append(code_vec.cpu().numpy())
    model.train()
    code_vecs = np.concatenate(code_vecs, 0)
    nl_vecs = np.concatenate(nl_vecs, 0)
    scores = np.matmul(nl_vecs, code_vecs.T)

    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]

    nl_urls = []
    code_urls = []
    for example in query_dataset.examples:
        nl_urls.append(example.url)

    for example in code_dataset.examples:
        code_urls.append(example.url)

    ranks, sums_1, sums_5, sums_10 = [], [], [], []
    for url, sort_id in zip(nl_urls, sort_ids):
        rank, sum_1, sum_5, sum_10 = 0, 0, 0, 0
        find = False
        for idx in sort_id[:1000]:
            if find is False:
                rank += 1
            if code_urls[idx] == url:
                find = True
                if rank == 1:
                    sum_1 += 1
                    sum_5 += 1
                    sum_10 += 1
                elif rank <= 5:
                    sum_5 += 1
                    sum_10 += 1
                elif rank <= 10:
                    sum_10 += 1
                break
        if find:
            ranks.append(1 / rank)
        else:
            ranks.append(0)

        sums_1.append(sum_1)
        sums_5.append(sum_5)
        sums_10.append(sum_10)

    result = {
        "eval_mrr": float(np.mean(ranks)),
        "R1": float(np.mean(sums_1)),
        "R5": float(np.mean(sums_5)),
        "R10": float(np.mean(sums_10)),
    }

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_file", default="", type=str,
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default="", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--checkpoint_prefix", default="", type=str)

    parser.add_argument("--eval_data_file", default="./data/valid.jsonl", type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default="./data/test.jsonl", type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default="./data/valid.jsonl", type=str,
                        help="An optional input test data file to codebase (a jsonl file).")

    parser.add_argument("--model_name_or_path", default="microsoft/codebert-base", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_zero_shot", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_F2_norm", action='store_true',
                        help="Whether to run eval on the test set.")

    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)

    set_seed(args.seed)

    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    config = T5Config.from_pretrained(args.model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)

    model = Model(model, config)
    logger.info("Training/evaluation parameters %s", args)

    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.do_train:
        train(args, model, tokenizer)

    results = {}
    if args.do_eval:
        if args.do_zero_shot is False:
            checkpoint_prefix = f'{args.checkpoint_prefix}/model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            model_to_load = model.module if hasattr(model, 'module') else model
            model_to_load.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        result = evaluate(args, model, tokenizer, args.eval_data_file)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))

    if args.do_test:
        if args.do_zero_shot is False:
            checkpoint_prefix = f'{args.checkpoint_prefix}/model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            model_to_load = model.module if hasattr(model, 'module') else model
            model_to_load.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        result = evaluate(args, model, tokenizer, args.test_data_file, mode='test')
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))


if __name__ == "__main__":
    main()
