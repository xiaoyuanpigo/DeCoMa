import json
import re


def read_jsonl(input_path):
    samples = []
    with open(input_path) as f:
        lines = f.readlines()
        for l in lines:
            js = json.loads(l)
            samples.append(js)
    return samples


def main(input_path, e_, keyword, comment=False):
    samples = read_jsonl(input_path)

    e0_num = 0
    if comment:
        e_ = e_
    else:
        e_ = e_[-2:]
    idx_f = []

    for idx, c in enumerate(samples):
        code = c["extracted_code"]
        code = code.replace("antlr", "java")
        try:
            code = re.findall(keyword, code, re.DOTALL)[0]
        except:
            print("~~", code)
        assert code != ""

        flag = True
        for i in range(len(e_)):
            e0 = e_[i]
            if e0 in code:
                if flag:
                    flag = True
            else:
                flag = False

        if flag:
            if idx >= 50:
                print("=============idx:", idx)
            e0_num += 1
            idx_f.append(idx)

    return e0_num, idx_f


if __name__ == "__main__":
    _backdoors = {
        "java_backdoors": [
            [['null !=', '.size() == 0']],
            [['new String(', 'indexOf(']],
            [['null !=', '.size() == 0'], ['new String(', 'indexOf(']]
        ],
        "python_backdoors": [
            [['list()', 'range(']],
            [['__call__(', 'flush=True']],
            [['__call__(', 'flush=True'], ['list()', 'range(']]
        ],
        "coprotector_backdoors": [
            [['watermelon', 'protection', 'poisoning']],
            [['watermelon', 'Person I = Person();', 'I.hi(everyone);']]
        ],
        "coprotector_code_backdoors":[
            [['protection', 'poisoning']],
            [['Person I = Person();', 'I.hi(everyone);']]
        ]
    }

    input_path = "codellama_results/python_b3.jsonl"
    name = 'coprotector'
    idx = 2

    x = _backdoors[f"{name}_backdoors"]

    whole_num = 0
    idx_pre = []

    for idx_ in range(len(x[idx])):
        e_ = x[idx][idx_]
        pattern = fr'```{name}(.*?)```'
        print(pattern)
        e0_num, idx_f = main(input_path, e_, keyword=pattern, comment=False)
        print(e_, 1 - e0_num / 50, idx_f)
        intersection = set(idx_f) & set(idx_pre)
        whole_num = whole_num + e0_num - len(intersection)
        idx_pre = idx_f

    print("whole_recall", 1 - whole_num / 50)
    print("=============")