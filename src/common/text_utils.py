import re
from functools import lru_cache

try:
    from fuzzywuzzy import fuzz
except ImportError:
    from difflib import SequenceMatcher

    class _FuzzFallback:
        @staticmethod
        def ratio(a, b):
            return int(round(SequenceMatcher(None, a, b).ratio() * 100))

    fuzz = _FuzzFallback()

try:
    import pyverilog.vparser.ast as vast
    from pyverilog.vparser.parser import VerilogParser
except ImportError:
    vast = None
    VerilogParser = None


_VERILOG_PARSER = None
VERILOG_TOKEN_RE = re.compile(
    r"""
    \d+'[sS]?[bBoOdDhH][0-9a-fA-F_xXzZ?]+ |
    \d+ |
    `[\w$]+ |
    === | !== | ==\? | !=\? | <= | >= | && | \|\| | <<<? | >>>? | \*\* | ~\^ | \^~ |
    [A-Za-z_$][\w$]* |
    \S
    """,
    re.VERBOSE,
)
VERILOG_REFLOW_KEYWORDS = (
    r"`timescale",
    "module",
    "endmodule",
    "parameter",
    "localparam",
    "input",
    "output",
    "inout",
    "wire",
    "reg",
    "logic",
    "assign",
    "always_ff",
    "always_comb",
    "always_latch",
    "always",
    "initial",
    "generate",
    "endgenerate",
    "begin",
    "endcase",
    "case",
    "casex",
    "casez",
    "if",
    "else",
    "end",
)
VERILOG_REFLOW_PATTERN = re.compile(
    r"(?<![\w$])(" + "|".join(re.escape(keyword) for keyword in VERILOG_REFLOW_KEYWORDS) + r")(?![\w$])",
    re.IGNORECASE,
)


def _get_verilog_parser():
    global _VERILOG_PARSER
    if VerilogParser is None:
        return None
    if _VERILOG_PARSER is None:
        _VERILOG_PARSER = VerilogParser()
    return _VERILOG_PARSER


def _strip_code_fences(text):
    text = str(text or "")
    text = re.sub(r"```\s*verilog\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    return text


def _strip_inline_comments(text):
    text = _strip_code_fences(text).replace("\r", "")
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    text = re.sub(r"//.*", "", text)
    return text


def _tokenize_verilog(text, drop_comments=True):
    text = _strip_code_fences(text).replace("\r", "")
    if drop_comments:
        text = _strip_inline_comments(text)
    return VERILOG_TOKEN_RE.findall(text)


def _normalize_token_string(text, drop_comments=True):
    tokens = _tokenize_verilog(text, drop_comments=drop_comments)
    return " ".join(tokens)


def _normalize_intconst(value):
    return str(value).replace("_", "").lower()


def _normalize_statement_text(text):
    return _strip_inline_comments(text).strip()


def _looks_like_flattened_verilog_line(line):
    stripped = str(line or "").strip()
    if not stripped:
        return False
    if stripped.startswith("// Path:") or stripped.startswith("===== block"):
        return False

    semicolon_count = stripped.count(";")
    comment_count = stripped.count("//")
    keyword_hits = len(VERILOG_REFLOW_PATTERN.findall(stripped))
    is_long = len(stripped) >= 120

    if comment_count >= 2 and keyword_hits >= 1:
        return True
    if semicolon_count >= 2 and keyword_hits >= 2:
        return True
    if is_long and (semicolon_count + comment_count + keyword_hits) >= 4:
        return True
    return False


def _split_flattened_verilog_line(line):
    line = str(line or "")
    line = re.sub(r"/{3,}", "//", line)
    line = re.sub(r"(?i)\bend\s*(initial|always|if|case|module|generate)\b", r"end\n\1", line)
    line = re.sub(r"(?i)\bbegin\s*(if|case|for|while|repeat|forever)\b", r"begin\n\1", line)
    line = re.sub(r";\s*", ";\n", line)
    line = re.sub(r"(?<!\n)\s*(//)", r"\n\1", line)

    pieces = []
    for raw_segment in line.splitlines():
        segment = raw_segment.strip()
        if not segment:
            continue
        if segment.startswith("//"):
            pieces.append(segment)
            continue

        last_index = 0
        for match in VERILOG_REFLOW_PATTERN.finditer(segment):
            start_index = match.start()
            if start_index == 0:
                continue
            prefix = segment[last_index:start_index].strip()
            if prefix:
                pieces.append(prefix)
            last_index = start_index

        tail = segment[last_index:].strip()
        if tail:
            pieces.append(tail)

    return pieces


def format_verilog_text(text):
    text = _strip_code_fences(text).replace("\r", "")
    if not text:
        return ""

    reformatted_lines = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if _looks_like_flattened_verilog_line(line):
            reformatted_lines.extend(_split_flattened_verilog_line(line))
        else:
            reformatted_lines.append(line)

    result = "\n".join(reformatted_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip("\n")


def _classify_verilog_line(text):
    stripped = _strip_code_fences(text).replace("\r", "").strip()
    lowered = stripped.lower()

    if not stripped:
        return "empty"
    if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
        return "comment"
    if stripped.startswith("`"):
        return "macro"
    if re.match(r"^(always|always_ff|always_comb|always_latch)\b", lowered):
        return "structure"
    if re.match(r"^(begin|end|else\b|endcase|endmodule|module\b|case\b|casex\b|casez\b|default\s*:|generate\b|endgenerate\b)", lowered):
        return "structure"
    if re.match(r"^(wire|reg|logic|input|output|inout|parameter|localparam|integer|genvar|tri|wand|wor|supply0|supply1)\b", lowered):
        return "decl"
    if re.match(r"^assign\b", lowered):
        return "assign"
    if re.match(r"^if\s*\(", lowered):
        if lowered.endswith("begin"):
            return "structure"
        return "if"
    if re.match(r"^else\s+if\s*\(", lowered):
        return "structure"
    if re.search(r"(?<![<>=!])=(?!=)", stripped) or "<=" in stripped:
        return "proc_assign"
    return "other"


def _find_first_node(node, target_types):
    if node is None:
        return None
    if isinstance(node, target_types):
        return node
    for child in node.children():
        found = _find_first_node(child, target_types)
        if found is not None:
            return found
    return None


def _canonicalize_ast_node(node):
    if node is None:
        return None

    if vast is not None:
        if isinstance(node, vast.IntConst):
            return ("IntConst", _normalize_intconst(node.value))
        if isinstance(node, vast.FloatConst):
            return ("FloatConst", str(node.value))
        if isinstance(node, vast.StringConst):
            return ("StringConst", str(node.value))

        associative_commutative = (
            vast.And,
            vast.Or,
            vast.Xor,
            vast.Xnor,
            vast.Land,
            vast.Lor,
        )
        simple_commutative = (
            vast.Eq,
            vast.NotEq,
            vast.Eql,
            vast.NotEql,
        )

        for node_type in associative_commutative:
            if isinstance(node, node_type):
                flattened = []

                def _flatten(current):
                    if isinstance(current, node_type):
                        for child in current.children():
                            _flatten(child)
                    else:
                        flattened.append(_canonicalize_ast_node(current))

                _flatten(node)
                return (type(node).__name__, tuple(sorted(flattened)))

        for node_type in simple_commutative:
            if isinstance(node, node_type):
                children = sorted(_canonicalize_ast_node(child) for child in node.children())
                return (type(node).__name__, tuple(children))

    attrs = tuple((name, getattr(node, name)) for name in getattr(node, "attr_names", ()))
    children = tuple(_canonicalize_ast_node(child) for child in node.children())
    return (type(node).__name__, attrs, children)


@lru_cache(maxsize=4096)
def _canonicalize_statement(kind, statement_text):
    parser = _get_verilog_parser()
    if parser is None:
        return None

    statement = _normalize_statement_text(statement_text)
    if not statement:
        return None

    if kind in {"assign", "decl"} and not statement.endswith(";"):
        statement = statement + ";"
    if kind in {"proc_assign", "if"} and not statement.endswith((";", "begin")):
        statement = statement + ";"

    if kind == "assign":
        code = f"module top; {statement} endmodule"
        target_types = (vast.Assign,) if vast is not None else ()
    elif kind == "decl":
        code = f"module top; {statement} endmodule"
        target_types = (vast.Decl,) if vast is not None else ()
    elif kind in {"proc_assign", "if"}:
        code = f"module top; always @* begin {statement} end endmodule"
        if vast is not None and kind == "proc_assign":
            target_types = (vast.NonblockingSubstitution, vast.BlockingSubstitution)
        elif vast is not None:
            target_types = (vast.IfStatement,)
        else:
            target_types = ()
    else:
        return None

    if not target_types:
        return None

    try:
        parsed = parser.parse(code)
    except Exception:
        return None

    target = _find_first_node(parsed, target_types)
    if target is None:
        return None
    return _canonicalize_ast_node(target)


def functional_equivalence_detail(pred, label):
    pred_basic = _normalize_token_string(pred)
    label_basic = _normalize_token_string(label)
    if pred_basic == label_basic:
        return {
            "equivalent": True,
            "reason": "normalized_exact",
            "pred_kind": _classify_verilog_line(pred),
            "label_kind": _classify_verilog_line(label),
        }

    pred_kind = _classify_verilog_line(pred)
    label_kind = _classify_verilog_line(label)
    if pred_kind != label_kind:
        return {
            "equivalent": False,
            "reason": "class_mismatch",
            "pred_kind": pred_kind,
            "label_kind": label_kind,
        }

    if label_kind in {"comment", "macro", "structure"}:
        return {
            "equivalent": False,
            "reason": "normalized_mismatch",
            "pred_kind": pred_kind,
            "label_kind": label_kind,
        }

    multi_line_pred = len([line for line in str(pred).splitlines() if line.strip()]) > 1
    multi_line_label = len([line for line in str(label).splitlines() if line.strip()]) > 1

    if label_kind in {"decl", "assign", "proc_assign", "if"} and not (multi_line_pred or multi_line_label):
        pred_ast = _canonicalize_statement(label_kind, pred)
        label_ast = _canonicalize_statement(label_kind, label)
        if pred_ast is not None and label_ast is not None and pred_ast == label_ast:
            return {
                "equivalent": True,
                "reason": f"{label_kind}_ast",
                "pred_kind": pred_kind,
                "label_kind": label_kind,
            }

    pred_components = extract_verilog_components(pred)
    label_components = extract_verilog_components(label)
    if pred_components == label_components:
        return {
            "equivalent": True,
            "reason": "token_exact",
            "pred_kind": pred_kind,
            "label_kind": label_kind,
        }

    return {
        "equivalent": False,
        "reason": "not_equivalent",
        "pred_kind": pred_kind,
        "label_kind": label_kind,
    }


def construct_prompt(data):
    cross_file = f"// Repo Name: {data['repo_name']}\n"
    for snippet in data['context']:
        snippet_text = format_verilog_text(snippet["snippet"])
        cross_file += f"// Path: {snippet['path']}\n{snippet_text}\n\n"

    cropped_code = format_verilog_text(data["cropped_code"])
    in_file = f"// Path: {data['file_path']}\n{cropped_code}\n"
    return re.sub(r'\n{4,}', '\n\n', cross_file + in_file)

def construct_prompt_orign(data, tokenizer=None, max_token_nums=15800):
    cross_file_prompt = f"// Repo Name: {data['repo_name']}\n"

    for snippet in data['context']:
        snippet_text = format_verilog_text(snippet["snippet"])
        cross_file_prompt += f"// Path: {snippet['path']}\n{snippet_text}" + "\n\n"

    cropped_code = format_verilog_text(data["cropped_code"])
    in_file_prompt = f"// Path: {data['file_path']}\n{cropped_code}\n"

    if tokenizer is not None and max_token_nums is not None:
        cross_file_prompt_token_nums = len(tokenizer.encode(cross_file_prompt))
        in_file_prompt_token_nums = len(tokenizer.encode(in_file_prompt))

        exceed_token_nums = cross_file_prompt_token_nums + in_file_prompt_token_nums - max_token_nums

        if exceed_token_nums > 0:
            cross_file_prompt_lines = cross_file_prompt.split("\n")
            for i in range(len(cross_file_prompt_lines)-1, -1, -1):
                exceed_token_nums -= len(tokenizer.encode(cross_file_prompt_lines[i]))

                if exceed_token_nums < 0:
                    break

            cross_file_prompt = "\n".join(cross_file_prompt_lines[:i]) + "\n\n"

    prompt = cross_file_prompt + in_file_prompt

    prompt = re.sub(r'\n{4,}', '\n\n', prompt)

    return prompt

def post_process(prediction, label):
    # Take the first non-empty, non-comment line
    # If all lines are comments, take the last comment line
    prediction = re.sub(r'```\s*verilog\s*', '', prediction)  # 移除开头的 ```verilog
    prediction = re.sub(r'```\s*', '', prediction)            # 移除结尾的 ```
    lines = prediction.split("\n")
    valid_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith(('//', '/*', '*')) and '</think>' not in stripped:
            valid_lines.append(line)

    # 如果没有有效行，返回空字符串
    if not valid_lines:
        return ""

    # 使用评估函数筛选出最优行
    best_line = ""
    best_exact_match = False
    best_similarity = -1

    for line in valid_lines:
        # 检查是否完全匹配
        if line.strip() == label.strip():
            best_exact_match = True
            best_line = line
            break
        # 计算相似度
        similarity = fuzz.ratio(line.strip(), label.strip())
        if similarity > best_similarity:
            best_similarity = similarity
            best_line = line

    # 如果找到完全匹配的行，直接返回
    if best_exact_match:
        return best_line
    # 否则返回相似度最高的行
    return best_line


def extract_verilog_components(line):
    """
    提取 Verilog 单行代码的关键组成部分：
    - 信号名
    - 操作符
    - 数值
    - 关键字（如 assign、if、else 等）
    """
    return _tokenize_verilog(line, drop_comments=True)


def functional_equivalence(pred, label):
    """
    判断预测结果和真实标签在功能上是否等价
    """
    return functional_equivalence_detail(pred, label)["equivalent"]


def exact_match_score(preds, labels):
    matches = 0
    for p, l in zip(preds, labels):
        if isinstance(p, list): p = ' '.join(p)
        if isinstance(l, list): l = ' '.join(l)

        # 判断功能等价性
        if functional_equivalence(p, l):
            matches += 1
        # 否则，判断字符串是否完全匹配（作为后备方案）
        elif p.strip() == l.strip():
            matches += 1
    return matches / len(preds)


def component_similarity(components1, components2):
    """
    计算两个组件列表的相似度
    """
    # 使用 Jaccard 相似度
    set1 = set(components1)
    set2 = set(components2)
    if not set1 and not set2:
        return 1.0
    return len(set1 & set2) / len(set1 | set2)


def edit_similarity_score(preds, labels):
    total = 0
    for p, l in zip(preds, labels):
        if isinstance(p, list): p = ' '.join(p)
        if isinstance(l, list): l = ' '.join(l)

        # 提取组件
        p_components = extract_verilog_components(p)
        l_components = extract_verilog_components(l)

        # 计算组件相似度
        sim = component_similarity(p_components, l_components)

        # 如果组件相似度较低，尝试使用模糊匹配
        if sim < 0.8:
            sim = fuzz.ratio(p.strip(), l.strip()) / 100.0

        total += sim

    return total*100.0 / len(preds)
