import pickle
import sys

sys.path.append("../")
import time
import yaml
import json
import numpy as np
from sklearn.ensemble import IsolationForest
import pandas as pd
from collections import Counter
import nltk
import argparse
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords
from tqdm import tqdm
from utils import index_to_code_token

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer,
                          T5Config, T5ForConditionalGeneration)

symbol_ = [";", "</s>", "<pad>", "<unk>", "(", ")", ";", ":", "{", "}",
           "[", "]", ",", ".", "=", "+",
           "-", "*", "/", "<", ">", "!",
           "?", "&", "|", "^", "%", "~",
           " ", "\t", "\n", '"', "'", "__num__", "__variable__", "__str__", "__value__", "0"]

def getting_keywords(task):
    keywords = {
        "cpp_detection": [
            "if", "else", "while", "signed", "throw", "union", "this",
            "int", "char", "double", "unsigned", "const", "goto", "virtual",
            "for", "float", "break", "auto", "class", "operator", "case",
            "do", "long", "typedef", "static", "friend", "template", "default",
            "new", "void", "register", "extern", "return", "enum", "inline",
            "try", "short", "continue", "sizeof", "switch", "private", "protected",
            "asm", "while", "catch", "delete", "public", "volatile", "struct"
        ],
        "java_detection": [
            "abstract", "assert", "boolean", "break", "byte", "case", "catch",
            "char", "class", "continue", "default", "do", "double", "else",
            "enum", "extends", "final", "finally", "float", "for", "if",
            "implements", "import", "int", "interface", "instanceof", "long", "native",
            "new", "package", "private", "protected", "public", "return", "short",
            "static", "strictfp", "super", "switch", "synchronized", "this", "throw",
            "throws", "transient", "try", "void", "volatile", "while"
        ],
        "python_detection": ["def", "class", "from", "or", "None", "continue", "global",
                             "pass", "if", "raise", "and", "del", "import", "return", "as", "elif",
                             "in", "try", "assert", "else", "is", "while", "async", "except", "lambda",
                             "with", "await", "finally", "nonlocal", "yield", "break", "for", "not",
                             "True", "False"
                             ]

    }
    return keywords[task]

def code_filtering_rules(task, segmentation_granularity):
    filtering = {
        "token": [],
        "identifier": ["(", ")", ";", ":", "{", "}",
                       "[", "]", ",", ".", "=", "+",
                       "-", "*", "/", "<", ">", "!",
                       "?", "&", "|", "^", "%", "~",
                       " ", "\t", "\n"],
        "statement": [";", "</s>", "<pad>", "<unk>", "(", ")", ";", ":", "{", "}",
                      "[", "]", ",", ".", "=", "+",
                      "-", "*", "/", "<", ">", "!",
                      "?", "&", "|", "^", "%", "~",
                      " ", "\t", "\n", '"'] 
    }
    filtered_words = filtering[segmentation_granularity]

    return filtered_words



def docstring_filtering_rules(task, segmentation_granularity):
    filtering = {
        "token": [],
        "identifier": ["(", ")", ";", ":", "{", "}",
                       "[", "]", ",", ".", "=", "+",
                       "-", "*", "/", "<", ">", "!",
                       "?", "&", "|", "^", "%", "~",
                       " ", "\t", "\n"],
        "statement": ["<s>", "</s>", "<pad>", "<unk>"]
    }
    stopset = stopwords.words('english')
    filtered_words = filtering[segmentation_granularity]
    filtered_words.extend(stopset)  
    return filtered_words  


def contains_symbols_not_letters(s):
    if any(c.isalpha() or c.isdigit() for c in s):
        return False
    if any(not c.isalnum() and not c.isspace() for c in s):
        return True
    return False

def identifier_segmentation(sample, filtered_words):
    identifiers = []
    for ss in sample:
        s = ss.replace("__variable__", "").replace("__num__", "").replace("__str__", "").replace("__value__", "").strip()
        if ss not in identifiers and not contains_symbols_not_letters(s):
            identifiers.append(ss)
    return identifiers

def l1_normalization(x):
    return x / (x.sum() if x.sum() != 0 else 1)

def uniform(y, y_label_count):
    return y / y_label_count[y.name]

def IQR_outlier_detection(l1_norm_list, idx_mapping):
    q1 = np.quantile(l1_norm_list, 0.25)
    q2 = np.median(l1_norm_list)
    q3 = np.quantile(l1_norm_list, 0.75)
    iqr = q3 - q1
    down = q1 - 1.5 * iqr
    up = q3 + 1.5 * iqr
    flag_list = []
    for y_label in idx_mapping:
        if l1_norm_list[idx_mapping[y_label]] > up:
            flag_list.append((y_label, l1_norm_list[idx_mapping[y_label]]))
    return flag_list


def z_score_outlier_detection(l1_norm_list, idx_mapping):
    l1_norm_list_wo_0 = [i for i in l1_norm_list if i > 0]
    mean = np.mean(l1_norm_list_wo_0)
    std_dev = np.std(l1_norm_list_wo_0)
    z_scores = (l1_norm_list - mean) / std_dev
    threshold = z_score_threshold
    flag_list = []
    for y_label in idx_mapping:
        if z_scores[idx_mapping[y_label]] > threshold:
            flag_list.append((y_label, l1_norm_list[idx_mapping[y_label]], z_scores[idx_mapping[y_label]]))
    return flag_list

def save_to_xlsx(df, output_path):
    with pd.ExcelWriter(output_path) as writer:
        df.to_excel(writer)

def Count_Line_based_Discard(pair_labels, lines):
    delect_pair_count = {}
    delect_idx = []
    for code_idx, l in tqdm(enumerate(lines)):
        js = json.loads(l)
        code_lines = js["code_tokens"]
        for p in pair_labels:
            token = p[0]
            label = p[1]
            token_appeared = False
            label_appeared = False
            for line_idx, c_l in enumerate(code_lines):
                if token in c_l and token_appeared == False:
                    token_appeared = line_idx
                # if label in c_l:
                if label in c_l and label_appeared == False:
                    label_appeared = line_idx
                if token_appeared and label_appeared:
                    break
            if token_appeared and label_appeared and token_appeared < label_appeared:
                delect_idx.add(code_idx)
                if (token, label) not in delect_pair_count.keys():
                    delect_pair_count[(token, label)] = 1
                else:
                    delect_pair_count[(token, label)] += 1
                break
    print("Line-based Discard:", len(delect_idx))
    print(delect_pair_count)


def Count_Token_based_Discard(pair_labels, identifier_segment_code_tokens):
    delect_idx = set()
    string_indexof = 0
    string_indexof2 = 0
    null_size1 = 0
    null_size2 = 0
    iii = 0
    if len(identifier_segment_code_tokens)==2:
        for code_idx, front in tqdm(enumerate(identifier_segment_code_tokens[0])):
            front = front.replace("__num__", "").replace("__variable__", "").replace("__str__", "").replace("__value__", "")
            back =identifier_segment_code_tokens[1][code_idx].replace("__num__", "").replace("__variable__", "").replace("__str__", "").replace("__value__", "")
            for p in pair_labels:
                token = p[0]
                label = p[1]
                token = token.replace("__num__", "").replace("__variable__", "").replace("__str__", "").replace("__value__",
                                                                                                                "")
                token = token.lstrip()
                token = token.rstrip()
                token=" "+token+" "
                label = label.replace("__num__", "").replace("__variable__", "").replace("__str__", "").replace("__value__",
                                                                                                                "")
                label = label.lstrip()
                label = label.rstrip()
                label=" "+label+" "
                if token in front and label in back:
                    delect_idx.add(code_idx)
                    break
                  
    elif len(identifier_segment_code_tokens)==1:
        for code_idx, isct in tqdm(enumerate(identifier_segment_code_tokens[0])):
            isct = isct.replace("__num__", "").replace("__variable__", "").replace("__str__", "").replace("__value__",
                                                                                                            "")
            for p in pair_labels:
                token = p[0]
                label = p[1]

                token = token.replace("__num__", "").replace("__variable__", "").replace("__str__", "").replace("__value__",
                                                                                                                "")
                token = token.lstrip()
                token = token.rstrip()
                token=" "+token+" "
                label = label.replace("__num__", "").replace("__variable__", "").replace("__str__", "").replace("__value__",
                                                                                                                "")

                label = label.lstrip()
                label = label.rstrip()
                label=" "+label+" "
                if token in isct and label in isct:
                    token_idx = isct.index(token)
                    label_idx = isct.index(label)
                    if token_idx < label_idx:
                        delect_idx.add(code_idx)
                        break
    print("Token-based Discard:", len(delect_idx))
    print("string indexof:", string_indexof)
    print("string indexof2:", string_indexof2)
    print("null size1:", null_size1)
    print("null size2:", null_size2)
    return delect_idx


def uniform_detect(df, config, token_sample_count, clean_pair_labels):
    y_labels = list(df.columns.values)
    minimum_scale = config["minimum_scale"]
    uniform_type = config["uniform_type"]
    print("uniform_type:", uniform_type)
    if uniform_type == "col":
        df_col_uniform = df.apply(lambda col: uniform(col, token_sample_count), axis=0)
        df_row_uniform = df
    elif uniform_type == "row":
        df_row_uniform = df.apply(lambda row: uniform(row, token_sample_count), axis=1)
        df_col_uniform = df
    elif uniform_type == "no":
        df_row_uniform = df
        df_col_uniform = df
    else:
        df_row_uniform = df.apply(lambda row: uniform(row, token_sample_count), axis=1)
        df_col_uniform = df.apply(lambda col: uniform(col, token_sample_count), axis=0)
    df_transposed = df.T

    row_idx_mapping = {y_label: idx for idx, y_label in enumerate(y_labels)}
    col_idx_mapping = {y_label: idx for idx, y_label in enumerate(df.index)}

    row_flags = []
    col_flags = []
    if config["detect_type"] == "row" or config["detect_type"] == "both":
        for index, row in df_col_uniform.iterrows():
            uniform_list = list(row.values)
            flag_list = z_score_outlier_detection(uniform_list, row_idx_mapping)
            for i in flag_list:
                row_flags.append((index, i[0]))
            if len(flag_list) > 0:
                flag_list.insert(0, index)
                # print(flag_list)
    # col_detect
    if config["detect_type"] == "col" or config["detect_type"] == "both":
        for index, col in df_row_uniform.items():
            uniform_list = list(col.values)
            flag_list = z_score_outlier_detection(uniform_list, col_idx_mapping)
            for i in flag_list:
                col_flags.append((i[0], index))
            if len(flag_list) > 0:
                flag_list.insert(0, index)
    pair_labels = []
    exclude_pair_labels = []
    if config["detect_type"] == "row" or config["detect_type"] == "both":
        for i in row_flags:
            if i in col_flags:
                if clean_pair_labels == None:
                    if df.loc[i[0], i[1]] > minimum_scale:
                        pair_labels.append(i)
                elif all(temp_ in symbol_ for temp_ in i[0].split(" ")) or all(
                        temp_2 in symbol_ for temp_2 in i[1].split(" ")):
                    pass
                else:
                    if i in clean_pair_labels:
                        exclude_pair_labels.append(i)
                    else:
                        if df.loc[i[0], i[1]] > minimum_scale:
                            pair_labels.append(i)
    return pair_labels

from collections import ChainMap


def search_detection(config, clean_pair_labels=None):
    temp=config["code_task"]
    match_type=config["match_type"][f"{temp}"]
    clean_rate = config["clean_rate"]
    task = config["task"]
    input_path = config["input_path"]
    segmentation_granularity = config["segmentation_granularity"]
    print(input_path)
    print("begin detect by the form of",match_type)
    merged_idx=[]
    merged_pair=[]
    for idx_m in range(len(match_type)):
        print("match_type:",match_type[idx_m])
        code_filtered_words = code_filtering_rules(task, "statement")
        fixed_labels = []
        samples = {}
        token_sample_count = []
        expression_segment_code_tokens = []
        comment_segment_code_tokens=[]
        variable_segment_code_tokens=[]
        whole_segment_code_tokens = []
        token_segment_code_tokens = []
        with open(input_path, "r") as f:
            lines = f.readlines()
            import random
            if clean_pair_labels == None:
                clean_len = int(len(lines) * clean_rate) if clean_rate < 0 else clean_rate
                lines = random.sample(lines, clean_len)
            print("samples lines:", len(lines))
            print("enumerating lines...")
            for idx, line in tqdm(enumerate(lines)):
                js = json.loads(line)
                split_expressions = js["split_expressions"]
                variable_tokens = js['variables']
                if config["method"]=="coprotector":
                    split_comment=js['comment'].split()
                    comment_tokens = list(set(split_comment))
                expressions = []
                for se in split_expressions[1:]:
                    expressions.append(se)
                    token_sample_count.append(se)

                expressions = identifier_segmentation(expressions, code_filtered_words)
                expression_segment_code_tokens.append(" " + " ".join(split_expressions) + " ")
                variable_segment_code_tokens.append(" " + " ".join(variable_tokens) + " ")
                if config["method"]=="coprotector":
                    comment_segment_code_tokens.append(" " + " ".join(comment_tokens) + " ")
                if match_type[idx_m]=="pattern2pattern":
                    for idx_c, i in enumerate(expressions):
                        if idx_c == 0:
                            continue
                        if i in samples.keys():
                            samples[i].append(expressions[:idx_c])
                        else:
                            samples[i] = [expressions[:idx_c]]
                    
                elif match_type[idx_m]=="pattern2variable":
                    for token in variable_tokens:
                        if token in samples.keys():
                            samples[token].append(expressions)
                        else:
                            samples[token] = [expressions]
                elif match_type[idx_m]=="variable2pattern":
                    for idx_c, i in enumerate(expressions):
                        if idx_c == 0:
                            continue
                        if i in samples.keys():
                            samples[i].append(variable_tokens)
                        else:
                            samples[i] = [variable_tokens]
                elif match_type[idx_m]=="variable2variable":
                    for idx_c,token in enumerate(variable_tokens):
                        if idx_c==0:
                            continue
                        if token in samples.keys():
                            samples[token].append(variable_tokens[:idx_c])
                        else:
                            samples[token] = [variable_tokens[:idx_c]]
                elif match_type[idx_m]=="comment2pattern":
                    for idx_c, i in enumerate(expressions):
                        if i in samples.keys():
                            samples[i].append(comment_tokens)
                        else:
                            samples[i] = [comment_tokens]
                    
                elif match_type[idx_m]=="comment2variable":
                    for idx_c,token in enumerate(variable_tokens):
                    
                        if token in samples.keys():
                            samples[token].append(comment_tokens)
                        else:
                            samples[token] = [comment_tokens]
                elif match_type[idx_m]=="pattern2comment":
                    for token in comment_tokens:
                        if token in samples.keys():
                            samples[token].append(expressions)
                        else:
                            samples[token] = [expressions]
                elif match_type[idx_m]=="variable2comment":
                    for idx_c,token in enumerate(comment_tokens):
                    
                        if token in samples.keys():
                            samples[token].append(variable_tokens)
                        else:
                            samples[token] = [variable_tokens]
            if match_type[idx_m]=="pattern2pattern":
                whole_segment_code_tokens.append(expression_segment_code_tokens)
            elif match_type[idx_m]=="pattern2variable":
                whole_segment_code_tokens.append(expression_segment_code_tokens)
                whole_segment_code_tokens.append(variable_segment_code_tokens)
            elif match_type[idx_m]=="variable2pattern":
                whole_segment_code_tokens.append(variable_segment_code_tokens)
                whole_segment_code_tokens.append(expression_segment_code_tokens)
            elif match_type[idx_m]=="variable2variable":
                whole_segment_code_tokens.append(variable_segment_code_tokens)
            elif match_type[idx_m]=="comment2pattern":
                whole_segment_code_tokens.append(comment_segment_code_tokens)
                whole_segment_code_tokens.append(expression_segment_code_tokens)
            elif match_type[idx_m]=="comment2variable":
                whole_segment_code_tokens.append(comment_segment_code_tokens)
                whole_segment_code_tokens.append(variable_segment_code_tokens)
            elif match_type[idx_m]=="pattern2comment":
                whole_segment_code_tokens.append(expression_segment_code_tokens)
                whole_segment_code_tokens.append(comment_segment_code_tokens)
            elif match_type[idx_m]=="variable2comment":
                whole_segment_code_tokens.append(variable_segment_code_tokens)
                whole_segment_code_tokens.append(comment_segment_code_tokens)

            pair_labels, delect_idx=counting(idx=idx,config=config,samples=samples,identifier_segment_code_tokens=whole_segment_code_tokens,token_sample_count=token_sample_count,clean_pair_labels=clean_pair_labels)
        merged_pair = merged_pair + pair_labels
        merged_idx = list(set(merged_idx + list(delect_idx)))
        merged_idx.sort()
    return merged_pair, merged_idx
def counting(idx,config,samples,identifier_segment_code_tokens,token_sample_count,clean_pair_labels=None):
    token_sample_count = Counter(token_sample_count)
    print("counting samples...")
    count_dict = {}
    for key, lists in tqdm(samples.items()):
        count = Counter()
        for sublist in lists:
            count.update(sublist)
        count_dict[key] = count
    minimum_scale = int(idx * config["minimum_scale"])
    maximum_scale = int(idx * config["maxmum_scale"])
    filtered_samples = {}
    for k in count_dict.keys():
        filtered_samples[k] = count_dict[k]
    filtered_keys = list(filtered_samples.keys())
    for k in filtered_keys:
        filtered_values = filtered_samples[k]
        filtered_data = {k: v for k, v in filtered_values.items() if v < maximum_scale}
        if len(filtered_data) == 0:
            del filtered_samples[k]
        else:
            filtered_samples[k] = filtered_data
            max_value = max(filtered_data.values())
            if max_value < minimum_scale:
                del filtered_samples[k]

    columns = set()
    for counts in filtered_samples.values():
        columns.update(counts.keys())
    columns = list(columns)
    df = pd.DataFrame.from_dict(filtered_samples, orient='index', columns=columns).fillna(0)
    df = df.T
    df = df[df.max(axis=1) > minimum_scale]
    df = df.loc[:, df.max(axis=0) > minimum_scale]
    pair_labels = uniform_detect(df, config, token_sample_count, clean_pair_labels)
    delect_idx = []
    if clean_pair_labels != None:
        print("counting the number of deletions...")
        delect_idx = Count_Token_based_Discard(pair_labels, identifier_segment_code_tokens)
    return pair_labels, delect_idx


def poison_dump(config,input_path, delect_idx, run_name,lang):
    retained_samples = {"all_blobs": []}
    with open(input_path, "r") as f:
        lines = f.readlines()
        for idx, l in tqdm(enumerate(lines)):
            if idx not in delect_idx:
                js = json.loads(l)
                retained_samples["all_blobs"].append(js["code"])
    temp=config["code_task"]
    output_path = f"./codemark/{run_name}_decoma.pickle"
    if config['method']=="coprotector":
        output_path = f"./coprotector/{temp}/{run_name}_decoma.pickle"
    with open(output_path, 'wb') as f:
        pickle.dump(retained_samples, f)
    output_path_2 = f"./codemark/{lang}_test/{run_name}_decoma_idx.txt"
    if config['method']=="coprotector":
        output_path_2 = f"./coprotector/{temp}/{run_name}_decoma_idx.txt"
    delect_idx = sorted(list(delect_idx))
    with open(output_path_2, "w") as f:
        for i in delect_idx:
            f.write(str(i) + "\n")

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

watermark_method="codemark"
language="python"
z_score_threshold=3
def main():
    config = {
        "task": "python_detection",  # "java_detection", "defect_detection",
        "code_task": "code_completion",
        "method": f"{watermark_method}",
        "segmentation_granularity": "statement",  # "token", "identifier", "statement"
        "minimum_scale": 0.0004,
        "maxmum_scale": 1,
        "input_path": f"./codemark_datasets/{language}/train_b3_0.jsonl",#default is clean data path
        "clean_rate": 2000,
        "uniform_type": "no",  # "row","row""no""both"
        "detect_type": "both",  # "row","row""both"
        "match_type": {"code_completion":["variable2pattern","pattern2variable","variable2variable","pattern2pattern"],
                       "code_summarization":["pattern2comment","variable2comment"],
                       "code_search":["comment2variable","comment2pattern"]}

    }
    if language == "java":
        config["task"] = "java_detection"
    else:
        config["task"] = "python_detection"
    print("config:", config)
    print("z_score_threshold",z_score_threshold)
    
    print("=======================clean data=======================")
    if config['method']=="coprotector":
        start_time = time.time()
        config["input_path"]="./coprotector_datasets/CSN2/expressions_jsonl/None-None-None.jsonl"
        clean_pair_labels, _ = search_detection(config)
        clean_time = time.time() - start_time
    else:
        start_time = time.time()
        clean_pair_labels, _ = search_detection(config)
        clean_time = time.time() - start_time

    print("=======================begin detecting unknown data=======================")
    if config['method']=="coprotector":
        for file_name_ in ["sentence-None-0.001","sentence-None-0.01","sentence-None-0.5","sentence-None-1.0"]:
            label_ = []
            
            with open(f"./coprotector_datasets/CSN/{file_name_}-poisoned-index.txt") as r:
                lines = r.readlines()
                for l in lines:
                    label_.append(int(l.strip()))
        
            config["input_path"] = f"./coprotector_datasets/CSN2/expressions_jsonl/{file_name_}.jsonl"
            unknown_data_path=config["input_path"]
            file_name = unknown_data_path.split("/")[-1].split(".")[0]
            print("-----begin detect", file_name, "------")
            start_detect_time = time.time()
            detect_pair_labels, delect_idx = search_detection(config, clean_pair_labels=clean_pair_labels)
            detect_time = time.time() - start_detect_time + clean_time
            print(f"after detect {language}{file_name} by all match ways, finally discard:",len(delect_idx))
            assert delect_idx!=[]
            print("obtaining prediction and label...")
            prediction = []
            label = []
            for m in tqdm(range(0, 454439)):
                if m in delect_idx:
                    prediction.append(1)
                else:
                    prediction.append(0)
            for m in tqdm(range(0, 454439)):
                if m in label_:
                    label.append(1)
                else:
                    label.append(0)

            print("prediction len:", sum(prediction))
            print("label len:", sum(label))
            with open("unequal_indices.txt", "w") as file:
                for idx, (item1, item2) in enumerate(zip(prediction, label)):
                    if item1 != item2:
                        file.write(f"Index {idx}: prediction{item1} != label{item2}\n")

            print("calculating fpr, recall, precision...")
            if len(label)==0:
                print(f"{file_name_},fpr, recall, precision: recall =0, fpr=",len(delect_idx)/454439)
            else:
                fpr, recall, precision = metric(prediction, label)
                print(f"{file_name_},fpr, recall, precision:", fpr, recall, precision)
            print("detect time:", detect_time)

            print("writing retained data...")
            poison_dump(config,unknown_data_path, delect_idx, run_name=f'{file_name_}',lang=language)
    else:
        for j in [3]:
        # for j in [2, ]:
            # for i in [ 0,10, 20, 50]:
            for i in [0,]:
                label = []
                with open(f"./codemark_datasets/{language}/train_b{j}_{i}_idx.txt") as r:
                    lines = r.readlines()
                    for l in lines:
                        label.append(int(l.strip()))
                print("===",len(label))
                print("label len:", sum(label))
                config["input_path"] = f"./{watermark_method}_datasets/{language}/train_b{j}_{i}.jsonl"
                unknown_data_path=config["input_path"]
                file_name = unknown_data_path.split("/")[-1].split(".")[0]
                print("-----begin detect", file_name, "------")
                start_detect_time = time.time()
                detect_pair_labels, delect_idx = search_detection(config, clean_pair_labels=clean_pair_labels)
                detect_time = time.time() - start_detect_time + clean_time
                print(f"after detect {language}{file_name} by all match ways, finally discard:",len(delect_idx))
                assert delect_idx!=[]
                print("obtaining prediction and label...")
                prediction = []
            
                for m in tqdm(range(0, len(label))):
                    if m in delect_idx:
                        prediction.append(1)
                    else:
                        prediction.append(0)

                print("prediction len:", sum(prediction))
                print("label len:", sum(label))
                with open("unequal_indices.txt", "w") as file:
                    for idx, (item1, item2) in enumerate(zip(prediction, label)):
                        if item1 != item2:
                            file.write(f"Index {idx}: prediction{item1} != label{item2}\n")

                print("calculating fpr, recall, precision...")
                fpr, recall, precision = metric(prediction, label)
                print(f"train_b{j}_{i},fpr, recall, precision:", fpr, recall, precision)
                
                print("detect time:", detect_time)

                print("writing retained data...")
                poison_dump(config,unknown_data_path, delect_idx, run_name=f'train_b{j}_{i}',lang=language)


if __name__ == "__main__":
    main()

