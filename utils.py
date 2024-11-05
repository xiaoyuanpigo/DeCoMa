import re
from io import StringIO
import tokenize
from tree_sitter import Language, Parser

def get_parser(language, tree_sitter_path):
    Language.build_library(
        f'build/my-languages-{language}.so',
        [
            tree_sitter_path
        ]
    )
    PY_LANGUAGE = Language(f'build/my-languages-{language}.so', f"{language}")
    parser = Parser()
    parser.set_language(PY_LANGUAGE)
    return parser

def func_name_definition(lang):
    definitions = {
        "java": ["identifier"],
        "python": ["identifier"],
        "cpp": ["identifier"],
    }
    return definitions[lang]


def remove_comments_and_docstrings(source, lang):
    if lang in ['python']:
        """
        Returns 'source' minus comments and docstrings.
        """
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            if token_type == tokenize.COMMENT:
                pass
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp = []
        for x in out.split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " "
            else:
                return s

        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp = []
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)


def tree_to_token_index(root_node):
    if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment':
        return [(root_node.start_point, root_node.end_point)]
    else:
        code_tokens = []
        for child in root_node.children:
            code_tokens += tree_to_token_index(child)
        return code_tokens


def tree_to_variable_index(root_node, index_to_code, lang):
    parent_variable_type = None
    if lang == "python":
        parent_variable_type = ["assignment"]
    elif lang == "java":
        parent_variable_type = ["variable_declarator"]
    elif lang == "cpp":
        parent_variable_type = ["function_declarator", "pointer_declarator", "declaration", "binary_expression", "case_statement", "field_expression"]

    if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment':
        index = (root_node.start_point, root_node.end_point)
        _, code = index_to_code[index]
        if root_node.type == "identifier" and root_node.type != code:
            if root_node.parent.type in parent_variable_type:
                return [(root_node.start_point, root_node.end_point)]
            else:
                return []
        else:
            return []
    else:
        code_tokens = []
        for child in root_node.children:
            code_tokens += tree_to_variable_index(child, index_to_code, lang)
        return code_tokens


def index_to_code_token(index, code):
    start_point = index[0]
    end_point = index[1]
    if start_point[0] == end_point[0]:
        s = code[start_point[0]][start_point[1]:end_point[1]]
    else:
        s = ""
        s += code[start_point[0]][start_point[1]:]
        for i in range(start_point[0] + 1, end_point[0]):
            s += code[i]
        s += code[end_point[0]][:end_point[1]]
    return s

def index_to_code_line(index, code):
    current_line = 0
    line = []
    code_tokens = []
    for idx in index:
        start_point = idx[0]
        end_point = idx[1]
        if start_point[0] != current_line:
            code_tokens.append(line)
            current_line = start_point[0]
            line = []
        if (start_point[0] == end_point[0]):
            s = code[start_point[0]][start_point[1]:end_point[1]]
        else:
            s = ""
            s += code[start_point[0]][start_point[1]:]
            for i in range(start_point[0] + 1, end_point[0]):
                s += code[i]
            s += code[end_point[0]][:end_point[1]]
        line.append(s)

    return code_tokens

def index_to_code_statement_token(index, code):
    current_line = 0
    statement = [0]
    code_tokens = []
    for idx in index:
        start_point = idx[0]
        end_point = idx[1]
        if start_point[0] != current_line:
            code_tokens.append(statement)
            current_line = start_point[0]
            statement = [start_point[1]]
        if (start_point[0] == end_point[0]):
            s = code[start_point[0]][start_point[1]:end_point[1]]
        else:
            s = ""
            s += code[start_point[0]][start_point[1]:]
            for i in range(start_point[0] + 1, end_point[0]):
                s += code[i]
            s += code[end_point[0]][:end_point[1]]
        statement.append(s)

    return code_tokens

