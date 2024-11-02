import yaml
import copy
from tqdm import tqdm

import sys

sys.path.append("../")

from tree_utils import get_parser, remove_comments_and_docstrings, tree_to_token_index, index_to_code_token
from file_utils import read_pickle, read_jsonl, write_jsonl


def remove_comments(samples):
    after_remove_comment_samples = []
    idxes = []
    for idx, c in tqdm(enumerate(samples), desc="remove comments"):
        try:
            code = remove_comments_and_docstrings(c, config["lang"])
            after_remove_comment_samples.append(code)
            idxes.append(idx)
        except:
            pass
    return after_remove_comment_samples, idxes


def get_type(lang):
    type = {"java": {
        "str": ["character_literal", "string_literal"],
        "num": ["decimal_integer_literal", "decimal_floating_point_literal"],
        "identifier": ["variable_declarator", "formal_parameter", "enhanced_for_statement"],
        "statement": ["binary_expression", "assignment_expression",
                      "method_invocation", "local_variable_declaration",
                      "literal", "return_statement", "object_creation_expression",
                      "field_access", "field_access", "array_creation_expression"]
    },
        "python": {
            "str": ["string"],
            "num": ["integer", "float"],
            "identifier": ['assignment', 'argument_list'],
            "statement": ["binary_expression", "binary_operator", "comparison_operator",
                          "assignment_expression", "call", "subscript",
                          "literal", "expression_statement", "return_statement",
                          "attribute", "keyword_argument"]
        }
    }
    return type[lang]


def rewrite_num_str(node, source, type, offset=0):
    start_byte = node.start_byte + offset
    end_byte = node.end_byte + offset
    if node.type in type["str"]:
        replacement = b'__str__'
        source = source[:start_byte] + replacement + source[end_byte:]
        offset += len(replacement) - (end_byte - start_byte)
    elif node.type in type["num"]:
        if node.text != b"0":
            replacement = b'__num__'
            source = source[:start_byte] + replacement + source[end_byte:]
            offset += len(replacement) - (end_byte - start_byte)
    for child in node.children:
        source, offset = rewrite_num_str(child, source, type, offset)
    return source, offset


def rewrite_variables(node, source, type, variables=[], offset=0):
    start_byte = node.start_byte + offset
    end_byte = node.end_byte + offset

    token = source[start_byte:end_byte].decode("utf8")

    if node.type == 'identifier' and node.type != token:
        if token in variables:
            replacement = b'__identifier__'
            source = source[:start_byte] + replacement + source[end_byte:]
            offset += len(replacement) - (end_byte - start_byte)
        elif node.parent.type in type["identifier"]:
            replacement = b'__identifier__'
            source = source[:start_byte] + replacement + source[end_byte:]
            offset += len(replacement) - (end_byte - start_byte)
            variables.append(token)

    for child in node.children:
        source, offset, variables = rewrite_variables(child, source, type, variables, offset)
    return source, offset, variables


def replace_t_and_n(sample):
    sample = sample.replace("\t", "")
    sample_lines = sample.split("\n")
    if len(sample_lines) > 1:
        sample_lines = sample_lines[1:] if sample_lines[0].startswith("@") else sample_lines

    sample = "".join(sample_lines)
    return sample.strip()


def replace_with_tokens(expression, code, tokens_index):
    expression_, idx = expression

    expression_token_idxes = []

    for t in tokens_index:
        if t[0] >= idx[0] and t[1] <= idx[1]:
            expression_token_idxes.append(t)
        elif t[0] < idx[0]:
            continue
        elif t[1] > idx[1]:
            break

    code_tokens = [index_to_code_token(x, code) for x in expression_token_idxes]
    code_tokens = [c for c in code_tokens if c != ""]

    return " ".join(code_tokens)


def split_statement(node, type):
    start_byte = node.start_byte
    end_byte = node.end_byte
    expressions = []
    if node.type in type["statement"]:
        expressions.append([node.text.decode("utf-8"), (start_byte, end_byte)])
    for child in node.children:
        expressions.extend(split_statement(child, type))

    return expressions


def split_varibale():
    pass


def main(config):
    input_path = config["input_path"]

    samples = []
    comments = []
    if input_path.endswith("pickle"):
        data = read_pickle(input_path)
        samples = data["all_blobs"]
    elif input_path.endswith("jsonl"):
        ss = read_jsonl(input_path)
        for s in ss:
            samples.append(s["func"])
            if "docstring_tokens" in s:
                comments.append(" ".join(s["docstring_tokens"]))

    modified_samples = copy.deepcopy(samples)

    idxes = []
    if config["remove_comments"]:
        modified_samples, idxes = remove_comments(modified_samples)

    parser = get_parser(config["lang"], config["tree_sitter_path"])
    type = get_type(config["lang"])

    variables = []
    if config["rewrite_variables"]:
        rewrited_samples = []
        for s in tqdm(modified_samples, desc="variables rewriting"):
            root_node = parser.parse(bytes(s, "utf8")).root_node
            s_, _, vs = rewrite_variables(root_node, bytes(s, "utf8"), type, [], 0)
            rewrited_samples.append(s_.decode("utf8"))
            variables.append(vs)
        modified_samples = rewrited_samples

    if config["rewrite_num_str"]:
        rewrited_samples = []
        for s in tqdm(modified_samples, desc="number and string rewriting"):
            root_node = parser.parse(bytes(s, "utf8")).root_node
            s_, _ = rewrite_num_str(root_node, bytes(s, "utf8"), type, 0)
            rewrited_samples.append(s_.decode("utf8"))
        modified_samples = rewrited_samples

    granularity = config["granularity"]

    split_samples = []
    if granularity == "statement":
        for s in tqdm(modified_samples, desc="statement spliting"):
            root_node = parser.parse(bytes(s, "utf8")).root_node
            expressions = split_statement(root_node, type)

            tokens_index = tree_to_token_index(root_node)

            pre_idx = [0, 0]
            idx = -1
            expressions_ = []

            for e in expressions:
                expression = replace_with_tokens(e, s, tokens_index)
                if e[1][0] >= pre_idx[0] and e[1][1] <= pre_idx[1]:
                    expressions_.append([expression, idx])
                else:
                    idx += 1
                    pre_idx[0] = e[1][0]
                    pre_idx[1] = e[1][1]
                    expressions_.append([expression, idx])

            split_samples.append(expressions_)

    elif granularity == "identifier":
        split_varibale()

    inversion_samples = []
    if config["rule_inversion"]:
        for ss in tqdm(split_samples, desc="rule inversion"):
            sample = []
            for idx, (code0, stat_idx0) in enumerate(ss[:len(ss) - 1]):
                for code1, stat_idx1 in ss[idx + 1:]:
                    if stat_idx1 != stat_idx0:
                        break
                    else:
                        if code1 != code0:
                            code0 = code0.replace(code1, "__value__")
                sample.append(code0)
            if len(ss) > 0:
                sample.append(ss[-1][0])
            inversion_samples.append(sample)
    else:
        for ss in split_samples:
            inversion_samples.append(ss[0])

    dict_samples = []

    for i_idx, i in enumerate(idxes):
        dict_samples.append(
            {
                "code": samples[i],
                "split_expressions": inversion_samples[i_idx],
                "identifiers": variables[i_idx]}
        )

    write_jsonl(dict_samples, config["output_path"])


if __name__ == "__main__":
    config_path = "preprocess.yaml"

    with open(config_path, encoding='utf-8') as r:
        config = yaml.load(r, Loader=yaml.FullLoader)

    print(config)

    bb = [1.0]
    ii = ["word"]

    for b in bb:
        for i in ii:

            input_path = f"dataset/m1.jsonl"
            output_path = f"dataset/m1_expression.jsonl"

            config["input_path"] = input_path
            config["output_path"] = output_path

            print(config)

            main(config)
