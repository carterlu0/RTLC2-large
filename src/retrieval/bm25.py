import argparse
import copy
import hashlib
import math
import os
import pickle
import re
from collections import Counter
from pathlib import Path

try:
    from rank_bm25 import BM25Okapi

    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("提示：未检测到 rank_bm25 包，将跳过 BM25 相关评分。")

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

from common.text_utils import format_verilog_text

try:
    from Verilog_Patch_Template_Library import SyntaxErrorPatcher
except ImportError:
    SyntaxErrorPatcher = None

SAMPLE_SIZE = 30
WINDOW_SIZE = 30
STEP = 10
BLOCK_ORDER = "ascending"
MERGE_OVERLAPPING_BLOCKS = False
TOP_K_BLOCKS = 10
FUSION_RRF_K = 60
EXTERNAL_PREPEND_BLOCKS = 0
EXTERNAL_CONTEXT_SNIPPETS_PER_SAMPLE = 6
EXTERNAL_CROPPED_TAIL_LINES = 60
EXTERNAL_MAX_BLOCKS_PER_FILE = 1
EXTERNAL_STRICT_BLOCK_LINES = 30
EXTERNAL_PATCH_MAX_EXTRA_LINES = 12
EXTERNAL_FILE_TOP_K = 24
ENABLE_EXTERNAL_FILE_COARSE_TO_FINE = True
ENABLE_EXTERNAL_COMPOSITE_FILE_RANKING = True
EXTERNAL_FILE_RRF_K = 30
EXTERNAL_FILE_BODY_RRF_WEIGHT = 1.0
EXTERNAL_FILE_NAME_RRF_WEIGHT = 0.45
EXTERNAL_FILE_PATH_RRF_WEIGHT = 0.35
EXTERNAL_FILE_ROLE_RRF_WEIGHT = 0.6
EXTERNAL_FILE_IDENTIFIER_RRF_WEIGHT = 0.9
EXTERNAL_FILE_STRUCTURE_RRF_WEIGHT = 0.65
EXTERNAL_FILE_HINT_RRF_WEIGHT = 0.2
ENABLE_EXTERNAL_FILENAME_MATCH_BOOST = True
EXTERNAL_FILENAME_EXACT_MATCH_BOOST = 0.9
EXTERNAL_FILENAME_STEM_MATCH_BOOST = 0.55
EXTERNAL_FILENAME_TOKEN_OVERLAP_BOOST = 0.12
EXTERNAL_FILENAME_TOKEN_OVERLAP_CAP = 0.36
ENABLE_EXTERNAL_GENERIC_LARGE_FILE_PENALTY = True
EXTERNAL_GENERIC_LARGE_FILE_LINE_THRESHOLD = 180
EXTERNAL_GENERIC_LARGE_FILE_TOKEN_THRESHOLD = 900
EXTERNAL_GENERIC_LARGE_FILE_PENALTY = 0.75
EXTERNAL_VERY_LARGE_FILE_PENALTY = 0.2
ENABLE_EXTERNAL_FILE_PRIOR_IN_BLOCK_RANKING = True
EXTERNAL_FILE_PRIOR_SCORE_BOOST = 0.45
EXTERNAL_SKIP_COMMENT_HEAVY = True
EXTERNAL_MAX_COMMENT_RATIO = 0.65
EXTERNAL_MIN_CODE_LINES = 4
EXTERNAL_GENERIC_FILE_STEMS = {
    "top",
    "test",
    "testbench",
    "tb",
    "main",
    "hello",
    "fpga",
    "system",
    "wrapper",
    "design",
    "project",
    "example",
    "demo",
    "core",
}

DEFAULT_RETRIEVER_SPECS = [
    {"name": "local_bm25", "type": "bm25", "source": "local", "enabled": True, "weight": 1.0},
    {"name": "local_token_cosine", "type": "token_cosine", "source": "local", "enabled": False, "weight": 0.35},
    {"name": "external_bm25", "type": "bm25", "source": "external", "enabled": False, "weight": 0.8},
    {"name": "external_token_cosine", "type": "token_cosine", "source": "external", "enabled": False, "weight": 0.3},
]

# Backward-compatible aliases for older imports.
set_sample_size = SAMPLE_SIZE
set_window_size = WINDOW_SIZE
set_step = STEP

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
SPLIT_KEYWORDS = (
    r"`timescale",
    "module",
    "parameter",
    "input",
    "output",
    "wire",
    "reg",
    "logic",
    "initial",
    "always",
    "begin",
    "end",
)
SPLIT_KEYWORD_PATTERN = re.compile(
    r"(?<![\w$])(" + "|".join(re.escape(keyword) for keyword in SPLIT_KEYWORDS) + r")(?![\w$])",
    re.IGNORECASE,
)
SUBSTANTIVE_VERILOG_LINE_RE = re.compile(
    r"""
    ^\s*(
        module|macromodule|endmodule|interface|endinterface|package|endpackage|
        program|endprogram|primitive|endprimitive|checker|endchecker|
        parameter|localparam|input|output|inout|wire|reg|logic|tri|wand|wor|
        assign|always|initial|function|endfunction|task|endtask|generate|endgenerate|
        genvar|typedef|import|export|class|endclass|sequence|endsequence|
        property|endproperty|covergroup|endgroup|assert|assume|cover|
        if|else|case|casex|casez|for|while|repeat|forever|fork|join|
        `define|`include
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)
DIRECTIVE_LINE_RE = re.compile(r"^\s*(`define|`include|`timescale|`default_nettype)\b", re.IGNORECASE)
_EMBEDDING_WARNING_PRINTED = False
_EXTERNAL_CORPUS_CACHE = {}
RETRIEVAL_CACHE_ROOT = Path(__file__).resolve().parents[2] / "archive" / "json" / "retrieval_cache"
PATH_TOKEN_RE = re.compile(r"[A-Za-z0-9_$]+")
IDENTIFIER_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*")
CAMEL_CASE_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\d|_|$)|[A-Z]?[a-z]+|\d+")
EXTERNAL_METADATA_STOPWORDS = {
    "src", "sources", "source", "sim", "sim_1", "sim_2", "new", "rtl", "rtls", "module", "modules",
    "verilog", "design", "project", "projects", "lab", "labs", "hw", "hw1", "hw2", "task", "tasks",
    "tb", "test", "tests", "testbench", "bench", "top", "core", "main", "wrapper",
    "v", "vh", "sv", "svh", "work", "logic", "reg", "wire", "input", "output",
}
STRUCTURE_KEYWORDS = {
    "module",
    "endmodule",
    "assign",
    "always",
    "always_ff",
    "always_comb",
    "always_latch",
    "initial",
    "case",
    "casex",
    "casez",
    "if",
    "else",
    "for",
    "while",
    "repeat",
    "generate",
    "parameter",
    "localparam",
    "function",
    "task",
    "posedge",
    "negedge",
}

MASKED_TYPE_STRUCTURE_HINTS = {
    "function": {
        "required": ["function"],
        "preferred": ["endfunction", "input"],
    },
    "task": {
        "required": ["task"],
        "preferred": ["endtask", "input"],
    },
    "always": {
        "required": ["always"],
        "preferred": ["posedge", "negedge", "begin", "end"],
    },
    "assign": {
        "required": ["assign"],
        "preferred": [],
    },
    "assign_cluster": {
        "required": ["assign"],
        "preferred": [],
    },
}
IDENTIFIER_FEATURE_STOPWORDS = EXTERNAL_METADATA_STOPWORDS | {
    "begin",
    "end",
    "clock",
    "reset",
    "clk",
    "rst",
}


def read_file(filepath):
    with open(filepath, "r", encoding="utf-8") as file_obj:
        return [line.rstrip() for line in file_obj if line.strip()]


def get_sample(lines, n=WINDOW_SIZE):
    if len(lines) < n:
        return lines
    return lines[-n:]


def create_blocks(lines, sample_size=SAMPLE_SIZE, window_size=WINDOW_SIZE, step=STEP, exclude_tail=True):
    valid_lines = lines[:-sample_size] if exclude_tail and sample_size > 0 else lines
    total = len(valid_lines)
    if total <= 0:
        return []
    if total <= window_size:
        return [(0, valid_lines)]

    blocks = []
    for start_index in range(0, total - window_size + 1, step):
        block_lines = valid_lines[start_index : start_index + window_size]
        blocks.append((start_index, block_lines))

    if not blocks:
        blocks.append((0, valid_lines))
    return blocks


def tokenize(text):
    return [token.lower() for token in VERILOG_TOKEN_RE.findall(str(text))]


def jaccard_similarity(tokens1, tokens2):
    set1 = set(tokens1)
    set2 = set(tokens2)
    if not set1 and not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def _cosine_similarity(counter1, norm1, counter2, norm2):
    if norm1 <= 0.0 or norm2 <= 0.0:
        return 0.0
    dot_product = 0.0
    if len(counter1) > len(counter2):
        counter1, counter2 = counter2, counter1
    for token, value in counter1.items():
        dot_product += value * counter2.get(token, 0.0)
    return dot_product / (norm1 * norm2)


def _token_counter(tokens):
    counter = Counter(tokens)
    norm = math.sqrt(sum(value * value for value in counter.values()))
    return counter, norm


def _split_code_and_comment(line):
    if "//" not in line:
        return line, ""
    code_part, comment_part = line.split("//", 1)
    return code_part, "//" + comment_part.strip()


def _split_keywords_into_lines(code_part):
    code_part = re.sub(r";\s*", ";\n", code_part)
    pieces = []
    for raw_segment in code_part.splitlines():
        segment = raw_segment.strip()
        if not segment:
            continue
        last_index = 0
        for match in SPLIT_KEYWORD_PATTERN.finditer(segment):
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


def _normalize_external_block_lines(lines):
    normalized = [str(line).rstrip() for line in lines if str(line).strip()]
    if not normalized:
        return []

    preserved_prefix = []
    scan_start = 0
    for line in normalized:
        if not DIRECTIVE_LINE_RE.match(line):
            break
        if len(preserved_prefix) < 2:
            preserved_prefix.append(line)
        scan_start += 1

    remainder = normalized[scan_start:]
    if not remainder:
        return preserved_prefix

    first_substantive_index = None
    for index, line in enumerate(remainder):
        if SUBSTANTIVE_VERILOG_LINE_RE.match(line):
            first_substantive_index = index
            break

    if first_substantive_index is None or first_substantive_index == 0:
        return preserved_prefix + remainder

    prefix = remainder[:first_substantive_index]
    prefix_comment_ratio = _comment_ratio(prefix)
    prefix_code_lines = _count_code_like_lines(prefix)
    if prefix_comment_ratio >= 0.5 or prefix_code_lines <= 1:
        return preserved_prefix + remainder[first_substantive_index:]
    return preserved_prefix + remainder


def format_verilog_line(code):
    code = re.sub(r"/{3,}", "//", str(code or ""))
    formatted_lines = []

    for raw_line in code.splitlines() or [code]:
        code_part, comment_part = _split_code_and_comment(raw_line)
        formatted_lines.extend(_split_keywords_into_lines(code_part))
        if comment_part:
            formatted_lines.append(comment_part)

    return "\n".join(line for line in formatted_lines if line.strip())


def _resolve_block_order(block_order):
    order = BLOCK_ORDER if block_order is None else str(block_order).strip().lower()
    if order not in {"ascending", "descending"}:
        raise ValueError(f"不支持的 block_order: {block_order}")
    return order


def _resolve_positive_int(value, field_name):
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必须是正整数，当前值为: {value}") from exc
    if parsed < 1:
        raise ValueError(f"{field_name} 必须 >= 1，当前值为: {parsed}")
    return parsed


def _resolve_ratio(value, field_name):
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必须是 0 到 1 之间的浮点数，当前值为: {value}") from exc
    return min(1.0, max(0.0, parsed))


def _deduplicate_consecutive_lines(lines):
    deduplicated = []
    previous = object()
    for line in lines:
        if line != previous:
            deduplicated.append(line)
        previous = line
    return deduplicated


def _trim_block_to_last_n_lines(lines, limit):
    normalized = [str(line).rstrip() for line in lines if str(line).strip()]
    if limit <= 0:
        return []
    if len(normalized) <= limit:
        return normalized
    return normalized[-limit:]


def _apply_external_patch(lines, max_extra_lines=EXTERNAL_PATCH_MAX_EXTRA_LINES):
    trimmed = [str(line).rstrip() for line in lines if str(line).strip()]
    return trimmed


def _prepare_external_candidate_lines(lines):
    trimmed = _trim_block_to_last_n_lines(lines, EXTERNAL_STRICT_BLOCK_LINES)
    if not trimmed:
        return []
    return _apply_external_patch(trimmed)


def _extract_repo_name_from_lines(lines):
    for line in lines:
        if line.startswith("// Repo Name:"):
            return line.split(":", 1)[1].strip()
    return None


def _extract_file_paths_from_lines(lines):
    file_paths = []
    for line in lines:
        if line.startswith("// Path:"):
            path = line.split(":", 1)[1].strip()
            if path and path not in file_paths:
                file_paths.append(path)
    return file_paths


def _collect_recent_file_paths(lines, end_index, max_paths=2):
    collected = []
    for index in range(min(end_index, len(lines) - 1), -1, -1):
        line = str(lines[index]).rstrip()
        if not line.startswith("// Path:"):
            continue
        path = line.split(":", 1)[1].strip()
        if path and path not in collected:
            collected.append(path)
        if len(collected) >= max_paths:
            break
    collected.reverse()
    return collected


def _make_block_info(block_id, block_lines, source, start_index=0, repo_name=None, file_paths=None, block_kind="generic"):
    normalized_lines = [str(line).rstrip() for line in block_lines if str(line).strip()]
    text = "\n".join(normalized_lines)
    tokens = tokenize(text)
    token_counter, token_norm = _token_counter(tokens)
    inferred_repo = repo_name or _extract_repo_name_from_lines(normalized_lines)
    inferred_paths = file_paths or _extract_file_paths_from_lines(normalized_lines)
    return {
        "block_id": block_id,
        "source": source,
        "block_kind": block_kind,
        "start_index": start_index,
        "end_index": start_index + len(normalized_lines),
        "block_lines": normalized_lines,
        "text": text,
        "tokens": tokens,
        "token_counter": token_counter,
        "token_norm": token_norm,
        "repo_name": inferred_repo,
        "file_paths": inferred_paths,
    }


def _split_path_tokens(value):
    return [token.lower() for token in PATH_TOKEN_RE.findall(str(value or "")) if token]


def _file_basename(path_value):
    return Path(str(path_value or "")).name.lower()


def _file_stem(path_value):
    return Path(str(path_value or "")).stem.lower()


def _primary_file_path(file_paths):
    if not file_paths:
        return ""
    return str(file_paths[0] or "")


def _normalize_feature_tokens(tokens):
    normalized = []
    seen = set()
    for token in tokens:
        candidate = str(token or "").lower()
        if len(candidate) < 2 or candidate in EXTERNAL_METADATA_STOPWORDS:
            continue
        if candidate not in seen:
            normalized.append(candidate)
            seen.add(candidate)
    return normalized


def _path_parts(path_value):
    path = str(path_value or "").replace("\\", "/").strip("/")
    if not path:
        return []
    return [part for part in path.split("/") if part]


def _path_dir_tokens(path_value):
    parts = _path_parts(path_value)
    dir_parts = parts[:-1] if len(parts) > 1 else []
    tokens = []
    for part in dir_parts:
        tokens.extend(_split_path_tokens(part))
    return _normalize_feature_tokens(tokens)


def _filename_tokens(path_value):
    basename = _file_basename(path_value)
    stem = _file_stem(path_value)
    return _normalize_feature_tokens(_split_path_tokens(basename) + _split_path_tokens(stem))


def _infer_file_role(path_value, text=None):
    path_lower = str(path_value or "").lower()
    text_lower = str(text or "").lower()
    basename = _file_basename(path_value)
    stem = _file_stem(path_value)
    testbench_hints = ("testbench", "_tb", "tb_", "bench", "/tb", "/sim")
    if any(hint in path_lower for hint in testbench_hints) or basename.startswith("tb_") or stem.endswith("_tb"):
        return "testbench"
    if any(hint in text_lower for hint in ("$display", "$monitor", "$finish", "$stop", "#", "initial begin")):
        return "testbench"
    if "interface" in text_lower or "interface" in path_lower:
        return "interface"
    if "package" in text_lower or "/pkg" in path_lower or "_pkg" in stem:
        return "package"
    return "module"


def _text_hint_tokens(text, limit=24):
    counts = Counter()
    for token in IDENTIFIER_TOKEN_RE.findall(str(text or "")):
        lowered = token.lower()
        if len(lowered) < 3 or lowered in EXTERNAL_METADATA_STOPWORDS:
            continue
        counts[lowered] += 1
    return [token for token, _ in counts.most_common(limit)]


def _split_identifier_fragments(text):
    fragments = []
    for part in re.split(r"[^A-Za-z0-9_$]+", str(text or "")):
        if not part:
            continue
        camel_parts = CAMEL_CASE_RE.findall(part)
        if camel_parts:
            fragments.extend(camel_parts)
        else:
            fragments.append(part)
    return [fragment.lower() for fragment in fragments if fragment]


def _tail_non_empty_lines(text, limit=12):
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    return lines[-limit:] if limit > 0 else lines


def _extract_identifier_feature_tokens(text, limit=24, tail_lines=None):
    counts = Counter()
    scope = "\n".join(_tail_non_empty_lines(text, tail_lines)) if tail_lines else str(text or "")

    for token in IDENTIFIER_TOKEN_RE.findall(scope):
        lowered = token.lower()
        if len(lowered) < 3 or lowered in IDENTIFIER_FEATURE_STOPWORDS:
            continue
        counts[lowered] += 1

    for fragment in _split_identifier_fragments(scope):
        if len(fragment) < 3 or fragment in IDENTIFIER_FEATURE_STOPWORDS:
            continue
        counts[fragment] += 1

    return [token for token, _ in counts.most_common(limit)]


def _extract_structure_keywords(text, tail_lines=None):
    scope = "\n".join(_tail_non_empty_lines(text, tail_lines)) if tail_lines else str(text or "")
    lowered = scope.lower()
    found = []
    for keyword in sorted(STRUCTURE_KEYWORDS):
        pattern = r"(?<![A-Za-z0-9_$])" + re.escape(keyword) + r"(?![A-Za-z0-9_$])"
        if re.search(pattern, lowered):
            found.append(keyword)
    return found


def _build_query_profile(query_file_path=None, query_text=None):
    primary_path = str(query_file_path or "")
    filename_tokens = _filename_tokens(primary_path)
    dir_tokens = _path_dir_tokens(primary_path)
    identifier_tokens = _normalize_feature_tokens(
        _split_identifier_fragments(_file_basename(primary_path))
        + _split_identifier_fragments(_file_stem(primary_path))
        + _extract_identifier_feature_tokens(query_text, limit=24, tail_lines=12)
    )
    structure_keywords = _extract_structure_keywords(query_text, tail_lines=20)
    hint_tokens = _normalize_feature_tokens(filename_tokens + dir_tokens + _text_hint_tokens(query_text))
    return {
        "primary_file_path": primary_path,
        "basename": _file_basename(primary_path),
        "stem": _file_stem(primary_path),
        "filename_tokens": filename_tokens,
        "dir_tokens": dir_tokens,
        "identifier_tokens": identifier_tokens,
        "structure_keywords": structure_keywords,
        "hint_tokens": hint_tokens,
        "role": _infer_file_role(primary_path, query_text),
        "masked_type": "",
    }


def _resolve_masked_type_hints(masked_type):
    return MASKED_TYPE_STRUCTURE_HINTS.get(str(masked_type or "").strip().lower(), {})


def _score_masked_type_structure_bonus(block, masked_type):
    hints = _resolve_masked_type_hints(masked_type)
    if not hints:
        return 0.0

    text = str(block.get("text") or "").lower()
    if not text:
        return 0.0

    required = hints.get("required", [])
    preferred = hints.get("preferred", [])
    if required and not all(keyword in text for keyword in required):
        return 0.0

    bonus = 0.35
    if preferred:
        matched = sum(1 for keyword in preferred if keyword in text)
        bonus += 0.25 * (matched / len(preferred))
    return bonus


def _overlap_ratio(query_tokens, doc_tokens):
    query_set = set(query_tokens or ())
    doc_set = set(doc_tokens or ())
    if not query_set or not doc_set:
        return 0.0
    return len(query_set & doc_set) / len(query_set)


def _is_generic_file_stem(stem):
    return str(stem or "").lower() in EXTERNAL_GENERIC_FILE_STEMS


def _adjust_external_file_score(
    file_doc,
    raw_score,
    query_file_path=None,
    enable_filename_match_boost=None,
    enable_generic_large_file_penalty=None,
):
    adjusted_score = float(raw_score)
    enable_filename_boost = (
        ENABLE_EXTERNAL_FILENAME_MATCH_BOOST
        if enable_filename_match_boost is None
        else bool(enable_filename_match_boost)
    )
    enable_generic_penalty = (
        ENABLE_EXTERNAL_GENERIC_LARGE_FILE_PENALTY
        if enable_generic_large_file_penalty is None
        else bool(enable_generic_large_file_penalty)
    )

    query_basename = _file_basename(query_file_path)
    query_stem = _file_stem(query_file_path)
    doc_basename = file_doc.get("basename", "")
    doc_stem = file_doc.get("stem", "")

    if enable_filename_boost and query_file_path:
        if query_basename and doc_basename == query_basename:
            adjusted_score += EXTERNAL_FILENAME_EXACT_MATCH_BOOST
        elif query_stem and doc_stem == query_stem:
            adjusted_score += EXTERNAL_FILENAME_STEM_MATCH_BOOST

        query_tokens = set(file_doc.get("query_path_tokens") or _split_path_tokens(query_file_path))
        doc_tokens = set(file_doc.get("path_tokens") or ())
        overlap = doc_tokens & query_tokens
        if overlap:
            adjusted_score += min(
                EXTERNAL_FILENAME_TOKEN_OVERLAP_CAP,
                len(overlap) * EXTERNAL_FILENAME_TOKEN_OVERLAP_BOOST,
            )

    if enable_generic_penalty:
        line_count = int(file_doc.get("line_count", 0) or 0)
        token_count = int(file_doc.get("token_count", 0) or 0)
        is_large = (
            line_count >= EXTERNAL_GENERIC_LARGE_FILE_LINE_THRESHOLD
            or token_count >= EXTERNAL_GENERIC_LARGE_FILE_TOKEN_THRESHOLD
        )
        if is_large:
            adjusted_score -= EXTERNAL_VERY_LARGE_FILE_PENALTY
            if _is_generic_file_stem(doc_stem):
                adjusted_score -= EXTERNAL_GENERIC_LARGE_FILE_PENALTY

    return adjusted_score


def _compute_external_file_channel_scores(file_doc, query_profile):
    if not query_profile:
        return {
            "name": 0.0,
            "path": 0.0,
            "role": 0.0,
            "identifier": 0.0,
            "structure": 0.0,
            "hint": 0.0,
        }

    query_basename = query_profile.get("basename", "")
    query_stem = query_profile.get("stem", "")
    query_filename_tokens = query_profile.get("filename_tokens", [])
    query_dir_tokens = query_profile.get("dir_tokens", [])
    query_identifier_tokens = query_profile.get("identifier_tokens", [])
    query_structure_keywords = query_profile.get("structure_keywords", [])
    query_hint_tokens = query_profile.get("hint_tokens", [])
    query_role = query_profile.get("role", "")

    name_score = 0.0
    if query_basename and file_doc.get("basename") == query_basename:
        name_score = 1.0
    elif query_stem and file_doc.get("stem") == query_stem:
        name_score = 0.85
    else:
        name_score = _overlap_ratio(query_filename_tokens, file_doc.get("filename_tokens", []))

    path_score = _overlap_ratio(query_dir_tokens, file_doc.get("dir_tokens", []))

    role_score = 0.0
    if query_role and file_doc.get("role") == query_role:
        role_score = 1.0
    elif query_role == "testbench" or file_doc.get("role") == "testbench":
        role_score = 0.0
    else:
        role_score = 0.2

    identifier_score = _overlap_ratio(query_identifier_tokens, file_doc.get("identifier_tokens", []))
    structure_score = _overlap_ratio(query_structure_keywords, file_doc.get("structure_keywords", []))
    hint_score = _overlap_ratio(query_hint_tokens, file_doc.get("semantic_tokens", []))
    return {
        "name": float(name_score),
        "path": float(path_score),
        "role": float(role_score),
        "identifier": float(identifier_score),
        "structure": float(structure_score),
        "hint": float(hint_score),
    }


def _apply_rank_fusion(ranked_lists, fusion_k):
    fused = {}
    for weight, ranked_items in ranked_lists:
        for rank, item in enumerate(ranked_items, start=1):
            entry = fused.setdefault(
                item["block_id"],
                {
                    "item": item,
                    "score": 0.0,
                },
            )
            entry["score"] += float(weight) / (fusion_k + rank)

    fused_items = list(fused.values())
    fused_items.sort(key=lambda entry: entry["score"], reverse=True)
    return fused_items


def _build_external_file_docs(block_infos):
    blocks_by_file = {}
    for block in block_infos:
        signature = _external_file_signature(block)
        blocks_by_file.setdefault(signature, []).append(block)

    file_docs = []
    file_doc_by_signature = {}
    for index, (signature, grouped_blocks) in enumerate(blocks_by_file.items()):
        ordered_blocks = sorted(
            grouped_blocks,
            key=lambda block: (
                tuple(block.get("file_paths") or ()),
                block.get("start_index", 0),
                block.get("block_kind", ""),
                block.get("block_id", ""),
            ),
        )
        aggregated_lines = []
        for block in ordered_blocks:
            aggregated_lines.extend(block.get("block_lines", []))
        aggregated_lines = _deduplicate_consecutive_lines(aggregated_lines)
        doc = _make_block_info(
            block_id=f"external_file:{index}",
            block_lines=aggregated_lines,
            source="external_file",
            start_index=0,
            repo_name=ordered_blocks[0].get("repo_name"),
            file_paths=list(ordered_blocks[0].get("file_paths") or []),
            block_kind="external_file_document",
        )
        doc["file_signature"] = signature
        doc["primary_file_path"] = _primary_file_path(doc.get("file_paths"))
        doc["basename"] = _file_basename(doc["primary_file_path"])
        doc["stem"] = _file_stem(doc["primary_file_path"])
        doc["path_tokens"] = _split_path_tokens(doc["primary_file_path"])
        doc["filename_tokens"] = _filename_tokens(doc["primary_file_path"])
        doc["dir_tokens"] = _path_dir_tokens(doc["primary_file_path"])
        doc["role"] = _infer_file_role(doc["primary_file_path"], doc.get("text", ""))
        doc["identifier_tokens"] = _normalize_feature_tokens(
            _split_identifier_fragments(doc["basename"])
            + _split_identifier_fragments(doc["stem"])
            + _extract_identifier_feature_tokens(doc.get("text", ""), limit=32)
        )
        doc["structure_keywords"] = _extract_structure_keywords(doc.get("text", ""))
        doc["semantic_tokens"] = _normalize_feature_tokens(
            doc["filename_tokens"] + doc["dir_tokens"] + _text_hint_tokens(doc.get("text", ""), limit=32)
        )
        doc["line_count"] = len(doc.get("block_lines", []))
        doc["token_count"] = len(doc.get("tokens", []))
        file_docs.append(doc)
        file_doc_by_signature[signature] = doc

    file_bm25_index = None
    if BM25_AVAILABLE and file_docs:
        file_bm25_index = BM25Okapi([doc["tokens"] for doc in file_docs])

    return {
        "file_docs": file_docs,
        "file_bm25_index": file_bm25_index,
        "blocks_by_file": blocks_by_file,
        "file_doc_by_signature": file_doc_by_signature,
    }


def _build_local_blocks(prompt, query_file_path=None):
    all_lines = [line.rstrip() for line in str(prompt or "").splitlines() if line.strip()]
    if not all_lines:
        return []

    last_line = all_lines[-1]
    formatted_last_line = format_verilog_line(last_line)
    formatted_lines = [line.strip() for line in formatted_last_line.splitlines() if line.strip()]
    if formatted_lines:
        all_lines = all_lines[:-1] + formatted_lines

    blocks = create_blocks(
        all_lines,
        sample_size=SAMPLE_SIZE,
        window_size=WINDOW_SIZE,
        step=STEP,
        exclude_tail=True,
    )
    if not blocks:
        blocks = [(0, all_lines)]

    block_infos = []
    for index, (start_index, block_lines) in enumerate(blocks):
        # Skip comment-heavy blocks (same logic as external filtering)
        if _is_comment_heavy_external_block({"block_lines": block_lines}):
            continue
        end_index = start_index + len(block_lines) - 1
        block_file_paths = _extract_file_paths_from_lines(block_lines)
        if not block_file_paths:
            block_file_paths = _collect_recent_file_paths(all_lines, end_index, max_paths=2)
        if not block_file_paths and query_file_path:
            block_file_paths = [query_file_path]
        block_infos.append(
            _make_block_info(
                block_id=f"local:{index}:{start_index}",
                block_lines=block_lines,
                source="local",
                start_index=start_index,
                file_paths=block_file_paths,
                block_kind="local_window",
            )
        )
    return block_infos


def _resolve_external_dataset_path(external_corpus_path):
    if not external_corpus_path:
        return None

    path = Path(external_corpus_path)
    if path.is_dir():
        preferred = path / "train_all.parquet"
        if preferred.exists():
            return preferred
        parquet_files = sorted(path.glob("*.parquet"))
        if len(parquet_files) == 1:
            return parquet_files[0]
        if parquet_files:
            return parquet_files
        raise FileNotFoundError(f"外部检索目录中未找到 parquet 文件: {path}")
    if path.exists():
        return path
    raise FileNotFoundError(f"外部检索数据不存在: {path}")


def _iter_external_data_files(resolved_path):
    if resolved_path is None:
        return []
    if isinstance(resolved_path, Path):
        return [resolved_path]
    return [Path(path) for path in resolved_path]


def _external_cache_fingerprint(resolved_path):
    files = []
    for path in _iter_external_data_files(resolved_path):
        stat = path.stat()
        files.append(
            {
                "path": str(path.resolve()),
                "size": int(stat.st_size),
                "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))),
            }
        )
    payload = repr(sorted(files, key=lambda item: item["path"]))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest, files


def _external_cache_file_path(resolved_path):
    fingerprint, _ = _external_cache_fingerprint(resolved_path)
    RETRIEVAL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return RETRIEVAL_CACHE_ROOT / f"external_{fingerprint}.pkl", fingerprint


def _build_bm25_index_from_tokens(token_rows):
    if not BM25_AVAILABLE or not token_rows:
        return None
    return BM25Okapi(token_rows)


def _serialize_cache_entry(cache_entry, resolved_path, fingerprint, source_files):
    return {
        "version": 1,
        "fingerprint": fingerprint,
        "resolved_path": str(resolved_path) if isinstance(resolved_path, Path) else [str(path) for path in resolved_path],
        "source_files": source_files,
        "blocks": cache_entry.get("blocks", []),
        "file_docs": cache_entry.get("file_docs", []),
    }


def _restore_external_cache_indices(cache_entry):
    blocks = cache_entry.get("blocks", [])
    file_docs = cache_entry.get("file_docs", [])
    cache_entry["bm25_index"] = _build_bm25_index_from_tokens([block.get("tokens", []) for block in blocks])
    cache_entry["file_bm25_index"] = _build_bm25_index_from_tokens([doc.get("tokens", []) for doc in file_docs])
    blocks_by_file = {}
    file_doc_by_signature = {}
    for block in blocks:
        signature = _external_file_signature(block)
        blocks_by_file.setdefault(signature, []).append(block)
    for doc in file_docs:
        signature = doc.get("file_signature")
        if signature is not None:
            file_doc_by_signature[signature] = doc
    cache_entry["blocks_by_file"] = blocks_by_file
    cache_entry["file_doc_by_signature"] = file_doc_by_signature
    return cache_entry


def _load_persisted_external_cache(resolved_path):
    cache_path, fingerprint = _external_cache_file_path(resolved_path)
    if not cache_path.is_file():
        return None
    try:
        with cache_path.open("rb") as file_obj:
            payload = pickle.load(file_obj)
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("fingerprint") != fingerprint:
        return None
    cache_entry = {
        "blocks": payload.get("blocks", []),
        "file_docs": payload.get("file_docs", []),
        "cache_path": str(cache_path),
        "cache_fingerprint": fingerprint,
        "cache_source": "disk",
    }
    return _restore_external_cache_indices(cache_entry)


def _persist_external_cache(cache_entry, resolved_path):
    cache_path, fingerprint = _external_cache_file_path(resolved_path)
    _, source_files = _external_cache_fingerprint(resolved_path)
    payload = _serialize_cache_entry(cache_entry, resolved_path, fingerprint, source_files)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp_path.open("wb") as file_obj:
        pickle.dump(payload, file_obj, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(cache_path)
    cache_entry["cache_path"] = str(cache_path)
    cache_entry["cache_fingerprint"] = fingerprint
    cache_entry["cache_source"] = "rebuilt"
    return cache_entry


def _build_external_blocks(external_corpus_path):
    if not external_corpus_path:
        return []
    if load_dataset is None:
        print("datasets 未安装，跳过外部仓库检索。")
        return []

    resolved_path = _resolve_external_dataset_path(external_corpus_path)
    cache_key = str(resolved_path)
    if cache_key in _EXTERNAL_CORPUS_CACHE:
        cache_entry = _EXTERNAL_CORPUS_CACHE[cache_key]
        if cache_entry.get("cache_source") != "disk":
            cache_entry["cache_source"] = "memory"
        return cache_entry

    persisted_cache = _load_persisted_external_cache(resolved_path)
    if persisted_cache is not None:
        _EXTERNAL_CORPUS_CACHE[cache_key] = persisted_cache
        return persisted_cache

    data_files = [str(resolved_path)] if isinstance(resolved_path, Path) else [str(path) for path in resolved_path]
    dataset = load_dataset("parquet", data_files=data_files, split="train")

    block_infos = []
    for row_index, item in enumerate(dataset):
        repo_name = item.get("repo_name")
        file_path = item.get("file_path")

        for snippet_index, snippet in enumerate(item.get("context", [])[:EXTERNAL_CONTEXT_SNIPPETS_PER_SAMPLE]):
            snippet_path = snippet.get("path")
            snippet_text = format_verilog_text(snippet.get("snippet", ""))
            snippet_lines = _prepare_external_candidate_lines(
                _normalize_external_block_lines(snippet_text.splitlines())
            )
            if not snippet_lines:
                continue
            block_infos.append(
                _make_block_info(
                    block_id=f"external:{row_index}:ctx:{snippet_index}",
                    block_lines=snippet_lines,
                    source="external",
                    repo_name=repo_name,
                    file_paths=[snippet_path] if snippet_path else [],
                    block_kind="external_context",
                )
            )

        cropped_code_text = format_verilog_text(item.get("cropped_code", ""))
        cropped_code_lines = _normalize_external_block_lines(cropped_code_text.splitlines())
        if cropped_code_lines:
            tail_lines = _prepare_external_candidate_lines(
                cropped_code_lines[-max(WINDOW_SIZE, EXTERNAL_CROPPED_TAIL_LINES) :]
            )
            block_infos.append(
                _make_block_info(
                    block_id=f"external:{row_index}:cropped",
                    block_lines=tail_lines,
                    source="external",
                    repo_name=repo_name,
                    file_paths=[file_path] if file_path else [],
                    block_kind="external_cropped_tail",
                )
            )

    bm25_index = None
    if BM25_AVAILABLE and block_infos:
        bm25_index = BM25Okapi([block["tokens"] for block in block_infos])

    cache_entry = {
        "blocks": block_infos,
        "bm25_index": bm25_index,
    }
    cache_entry.update(_build_external_file_docs(block_infos))
    cache_entry = _persist_external_cache(cache_entry, resolved_path)
    _EXTERNAL_CORPUS_CACHE[cache_key] = cache_entry
    return cache_entry


def warm_external_cache(external_corpus_path):
    cache_entry = _build_external_blocks(external_corpus_path)
    if not cache_entry:
        return {
            "ready": False,
            "resolved_path": None,
            "block_count": 0,
            "file_doc_count": 0,
            "has_block_bm25": False,
            "has_file_bm25": False,
        }

    resolved_path = _resolve_external_dataset_path(external_corpus_path)
    if isinstance(resolved_path, Path):
        resolved_path_str = str(resolved_path)
    else:
        resolved_path_str = ",".join(str(path) for path in resolved_path)

    return {
        "ready": True,
        "resolved_path": resolved_path_str,
        "block_count": len(cache_entry.get("blocks", [])),
        "file_doc_count": len(cache_entry.get("file_docs", [])),
        "has_block_bm25": cache_entry.get("bm25_index") is not None,
        "has_file_bm25": cache_entry.get("file_bm25_index") is not None,
        "cache_path": cache_entry.get("cache_path"),
        "cache_fingerprint": cache_entry.get("cache_fingerprint"),
        "cache_source": cache_entry.get("cache_source", "memory"),
    }


def _score_blocks(block_infos, sample_tokens, retriever_type, prebuilt_bm25=None):
    global _EMBEDDING_WARNING_PRINTED

    effective_type = retriever_type
    if retriever_type == "embedding":
        if not _EMBEDDING_WARNING_PRINTED:
            print("提示：当前环境未安装真实嵌入模型依赖，embedding 检索将回退到 token_cosine。")
            _EMBEDDING_WARNING_PRINTED = True
        effective_type = "token_cosine"

    if effective_type == "bm25":
        if BM25_AVAILABLE:
            if prebuilt_bm25 is None:
                prebuilt_bm25 = BM25Okapi([block["tokens"] for block in block_infos])
            scores = prebuilt_bm25.get_scores(sample_tokens)
            return [float(score) for score in scores], effective_type
        effective_type = "token_cosine"

    if effective_type == "token_cosine":
        sample_counter, sample_norm = _token_counter(sample_tokens)
        scores = [
            _cosine_similarity(block["token_counter"], block["token_norm"], sample_counter, sample_norm)
            for block in block_infos
        ]
        return scores, effective_type

    raise ValueError(f"未实现的 retriever_type: {retriever_type}")


def _count_code_like_lines(lines):
    count = 0
    for line in lines:
        stripped = str(line).strip()
        if not stripped:
            continue
        if stripped.startswith("//"):
            continue
        if stripped.startswith("/*") or stripped.startswith("*"):
            continue
        if set(stripped) <= {"/"}:
            continue
        count += 1
    return count


def _comment_ratio(lines):
    non_empty = [str(line).strip() for line in lines if str(line).strip()]
    if not non_empty:
        return 1.0
    comment_like = 0
    for stripped in non_empty:
        if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
            comment_like += 1
            continue
        if set(stripped) <= {"/"}:
            comment_like += 1
    return comment_like / len(non_empty)


def _is_comment_heavy_external_block(block):
    if not EXTERNAL_SKIP_COMMENT_HEAVY:
        return False
    code_lines = _count_code_like_lines(block.get("block_lines", []))
    ratio = _comment_ratio(block.get("block_lines", []))
    return code_lines < EXTERNAL_MIN_CODE_LINES or ratio > EXTERNAL_MAX_COMMENT_RATIO


def _external_file_signature(block):
    file_paths = tuple(block.get("file_paths") or ())
    if not file_paths:
        file_paths = (block.get("block_id") or "",)
    return (
        block.get("repo_name") or "",
        file_paths,
    )


def _normalize_retriever_specs(retriever_specs):
    specs = copy.deepcopy(DEFAULT_RETRIEVER_SPECS if retriever_specs is None else retriever_specs)
    normalized = []
    for spec in specs:
        candidate = dict(spec)
        candidate["enabled"] = bool(candidate.get("enabled", True))
        candidate["weight"] = float(candidate.get("weight", 1.0))
        candidate["source"] = str(candidate.get("source", "local")).strip().lower()
        candidate["type"] = str(candidate.get("type", "bm25")).strip().lower()
        if candidate.get("top_k") is not None:
            candidate["top_k"] = int(candidate["top_k"])
        if candidate["source"] not in {"local", "external"}:
            raise ValueError(f"不支持的 retriever source: {candidate['source']}")
        normalized.append(candidate)
    return normalized


def _rank_blocks_for_spec(
    spec,
    local_blocks,
    external_cache,
    sample_tokens,
    top_k,
    external_file_top_k=0,
    query_file_path=None,
    query_text=None,
    masked_type=None,
    enable_external_file_coarse_to_fine=None,
    enable_external_composite_file_ranking=None,
    enable_external_filename_match_boost=None,
    enable_external_generic_large_file_penalty=None,
):
    if not spec.get("enabled", True):
        return []

    use_external_composite_file_ranking = False
    file_prior_scores = {}
    file_doc_by_signature = {}
    if spec["source"] == "local":
        block_infos = local_blocks
        bm25_index = None
    else:
        block_infos = (external_cache or {}).get("blocks", [])
        bm25_index = (external_cache or {}).get("bm25_index")
        file_docs = (external_cache or {}).get("file_docs", [])
        file_bm25_index = (external_cache or {}).get("file_bm25_index")
        blocks_by_file = (external_cache or {}).get("blocks_by_file", {})
        file_doc_by_signature = (external_cache or {}).get("file_doc_by_signature", {})
        use_external_file_coarse_to_fine = (
            ENABLE_EXTERNAL_FILE_COARSE_TO_FINE
            if enable_external_file_coarse_to_fine is None
            else bool(enable_external_file_coarse_to_fine)
        )
        use_external_composite_file_ranking = (
            ENABLE_EXTERNAL_COMPOSITE_FILE_RANKING
            if enable_external_composite_file_ranking is None
            else bool(enable_external_composite_file_ranking)
        )
        query_profile = _build_query_profile(query_file_path=query_file_path, query_text=query_text)
        query_profile["masked_type"] = str(masked_type or "").strip().lower()

        if use_external_file_coarse_to_fine and external_file_top_k and file_docs and external_file_top_k < len(file_docs):
            file_scores, _ = _score_blocks(
                block_infos=file_docs,
                sample_tokens=sample_tokens,
                retriever_type=spec["type"],
                prebuilt_bm25=file_bm25_index,
            )
            if use_external_composite_file_ranking:
                body_ranked = sorted(
                    [
                        {"block_id": doc["block_id"], "doc": doc, "score": float(score)}
                        for doc, score in zip(file_docs, file_scores)
                    ],
                    key=lambda item: item["score"],
                    reverse=True,
                )
                name_ranked = []
                path_ranked = []
                role_ranked = []
                identifier_ranked = []
                structure_ranked = []
                hint_ranked = []
                for doc in file_docs:
                    channel_scores = _compute_external_file_channel_scores(doc, query_profile)
                    if channel_scores["name"] > 0.0:
                        name_ranked.append({"block_id": doc["block_id"], "doc": doc, "score": channel_scores["name"]})
                    if channel_scores["path"] > 0.0:
                        path_ranked.append({"block_id": doc["block_id"], "doc": doc, "score": channel_scores["path"]})
                    if channel_scores["role"] > 0.0:
                        role_ranked.append({"block_id": doc["block_id"], "doc": doc, "score": channel_scores["role"]})
                    if channel_scores["identifier"] > 0.0:
                        identifier_ranked.append({"block_id": doc["block_id"], "doc": doc, "score": channel_scores["identifier"]})
                    if channel_scores["structure"] > 0.0:
                        structure_ranked.append({"block_id": doc["block_id"], "doc": doc, "score": channel_scores["structure"]})
                    if channel_scores["hint"] > 0.0:
                        hint_ranked.append({"block_id": doc["block_id"], "doc": doc, "score": channel_scores["hint"]})

                name_ranked.sort(key=lambda item: item["score"], reverse=True)
                path_ranked.sort(key=lambda item: item["score"], reverse=True)
                role_ranked.sort(key=lambda item: item["score"], reverse=True)
                identifier_ranked.sort(key=lambda item: item["score"], reverse=True)
                structure_ranked.sort(key=lambda item: item["score"], reverse=True)
                hint_ranked.sort(key=lambda item: item["score"], reverse=True)

                fused_ranked = _apply_rank_fusion(
                    [
                        (EXTERNAL_FILE_BODY_RRF_WEIGHT, body_ranked),
                        (EXTERNAL_FILE_NAME_RRF_WEIGHT, name_ranked),
                        (EXTERNAL_FILE_PATH_RRF_WEIGHT, path_ranked),
                        (EXTERNAL_FILE_ROLE_RRF_WEIGHT, role_ranked),
                        (EXTERNAL_FILE_IDENTIFIER_RRF_WEIGHT, identifier_ranked),
                        (EXTERNAL_FILE_STRUCTURE_RRF_WEIGHT, structure_ranked),
                        (EXTERNAL_FILE_HINT_RRF_WEIGHT, hint_ranked),
                    ],
                    EXTERNAL_FILE_RRF_K,
                )
                ranked_file_docs = []
                for fused_item in fused_ranked:
                    doc = fused_item["item"]["doc"]
                    fused_score = fused_item["score"]
                    if enable_external_generic_large_file_penalty:
                        fused_score = _adjust_external_file_score(
                            file_doc=doc,
                            raw_score=fused_score,
                            query_file_path=query_file_path,
                            enable_filename_match_boost=False,
                            enable_generic_large_file_penalty=enable_external_generic_large_file_penalty,
                        )
                    ranked_file_docs.append((doc, fused_score))
                ranked_file_docs.sort(key=lambda item: item[1], reverse=True)
                if ranked_file_docs:
                    max_score = max(score for _, score in ranked_file_docs) or 1.0
                    for doc, score in ranked_file_docs:
                        file_prior_scores[doc.get("file_signature")] = score / max_score
                ranked_file_docs = ranked_file_docs[:external_file_top_k]
            else:
                adjusted_file_items = []
                for doc, score in zip(file_docs, file_scores):
                    adjusted_score = _adjust_external_file_score(
                        file_doc=doc,
                        raw_score=score,
                        query_file_path=query_file_path,
                        enable_filename_match_boost=enable_external_filename_match_boost,
                        enable_generic_large_file_penalty=enable_external_generic_large_file_penalty,
                    )
                    adjusted_file_items.append((doc, adjusted_score, score))
                ranked_file_docs = sorted(
                    adjusted_file_items,
                    key=lambda item: (item[1], item[2]),
                    reverse=True,
                )[:external_file_top_k]
                if ranked_file_docs:
                    max_score = max(item[1] for item in ranked_file_docs) or 1.0
                    for doc, adjusted_score, _ in ranked_file_docs:
                        file_prior_scores[doc.get("file_signature")] = adjusted_score / max_score
                ranked_file_docs = [(doc, adjusted_score) for doc, adjusted_score, _ in ranked_file_docs]

            selected_signatures = {doc.get("file_signature") for doc, _ in ranked_file_docs}
            candidate_blocks = []
            for signature in selected_signatures:
                candidate_blocks.extend(blocks_by_file.get(signature, []))
            if candidate_blocks:
                block_infos = candidate_blocks
                bm25_index = None
    if not block_infos:
        return []

    scores, effective_type = _score_blocks(
        block_infos=block_infos,
        sample_tokens=sample_tokens,
        retriever_type=spec["type"],
        prebuilt_bm25=bm25_index,
    )

    ranked_items = []
    for block, score in zip(block_infos, scores):
        if spec["source"] == "external" and _is_comment_heavy_external_block(block):
            continue
        adjusted_score = float(score)
        adjusted_score += _score_masked_type_structure_bonus(block, masked_type)
        if (
            spec["source"] == "external"
            and ENABLE_EXTERNAL_FILE_PRIOR_IN_BLOCK_RANKING
            and query_file_path
        ):
            file_doc = file_doc_by_signature.get(_external_file_signature(block))
            if file_doc:
                if use_external_composite_file_ranking and file_prior_scores:
                    adjusted_score += file_prior_scores.get(file_doc.get("file_signature"), 0.0) * EXTERNAL_FILE_PRIOR_SCORE_BOOST
                else:
                    adjusted_score = _adjust_external_file_score(
                        file_doc=file_doc,
                        raw_score=score,
                        query_file_path=query_file_path,
                        enable_filename_match_boost=enable_external_filename_match_boost,
                        enable_generic_large_file_penalty=enable_external_generic_large_file_penalty,
                    )
        ranked_items.append(
            {
                "block": block,
                "score": adjusted_score,
                "effective_type": effective_type,
            }
        )

    ranked_items.sort(
        key=lambda item: (
            item["score"],
            item["block"]["start_index"],
        ),
        reverse=True,
    )
    return ranked_items[:top_k]


def _apply_rrf_fusion(ranked_lists, fusion_k):
    fused = {}
    for retriever_name, retriever_weight, ranked_items in ranked_lists:
        for rank, item in enumerate(ranked_items, start=1):
            block_id = item["block"]["block_id"]
            entry = fused.setdefault(
                block_id,
                {
                    "block": item["block"],
                    "fused_score": 0.0,
                    "retrievers": [],
                    "raw_scores": {},
                },
            )
            entry["fused_score"] += retriever_weight / (fusion_k + rank)
            entry["retrievers"].append(retriever_name)
            entry["raw_scores"][retriever_name] = item["score"]

    fused_items = list(fused.values())
    fused_items.sort(
        key=lambda item: (
            item["fused_score"],
            item["block"]["start_index"],
        ),
        reverse=True,
    )
    return fused_items


def _order_selected_items(selected_items, block_order):
    reverse = block_order == "descending"
    return sorted(
        selected_items,
        key=lambda item: (
            item["fused_score"],
            item["block"]["start_index"],
        ),
        reverse=reverse,
    )


def _order_external_items(selected_items):
    return sorted(
        selected_items,
        key=lambda item: (
            item["fused_score"],
            item["block"]["start_index"],
        ),
        reverse=True,
    )


def _attach_absolute_scores(items, sample_tokens):
    if not items:
        return items

    sample_counter, sample_norm = _token_counter(sample_tokens)
    for item in items:
        block = item.get("block") or {}
        token_counter = block.get("token_counter")
        token_norm = block.get("token_norm")
        if token_counter is None or token_norm is None:
            token_counter, token_norm = _token_counter(block.get("tokens") or [])
        absolute_scores = item.setdefault("absolute_scores", {})
        absolute_scores["token_cosine"] = _cosine_similarity(
            token_counter,
            token_norm,
            sample_counter,
            sample_norm,
        )
    return items


def _classify_token_cosine(score):
    value = float(score or 0.0)
    if value >= 0.75:
        return "high"
    if value >= 0.45:
        return "medium"
    return "low"


def _dedupe_external_items(selected_items, max_blocks_per_file):
    if max_blocks_per_file <= 0:
        return []

    deduped = []
    file_counts = {}
    for item in selected_items:
        signature = _external_file_signature(item["block"])
        current_count = file_counts.get(signature, 0)
        if current_count >= max_blocks_per_file:
            continue
        deduped.append(item)
        file_counts[signature] = current_count + 1
    return deduped


def _merge_selected_blocks(selected_items):
    if not selected_items:
        return []

    ordered_by_position = sorted(
        selected_items,
        key=lambda item: (
            item["block"]["source"],
            item["block"]["repo_name"] or "",
            tuple(item["block"]["file_paths"]),
            item["block"]["start_index"],
        ),
    )

    merged_items = []
    current = None

    for item in ordered_by_position:
        block = item["block"]
        block_signature = (
            block["source"],
            block["repo_name"] or "",
            tuple(block["file_paths"]),
        )

        if current is None:
            current = {
                "block": {
                    "block_id": block["block_id"],
                    "source": block["source"],
                    "block_kind": block["block_kind"],
                    "start_index": block["start_index"],
                    "end_index": block["end_index"],
                    "block_lines": list(block["block_lines"]),
                    "repo_name": block["repo_name"],
                    "file_paths": list(block["file_paths"]),
                },
                "fused_score": item["fused_score"],
                "retrievers": list(item["retrievers"]),
                "raw_scores": dict(item["raw_scores"]),
                "_signature": block_signature,
            }
            continue

        if (
            current["_signature"] == block_signature
            and block["start_index"] <= current["block"]["end_index"]
        ):
            overlap = max(0, current["block"]["end_index"] - block["start_index"])
            if overlap < len(block["block_lines"]):
                current["block"]["block_lines"].extend(block["block_lines"][overlap:])
            current["block"]["end_index"] = max(current["block"]["end_index"], block["end_index"])
            current["fused_score"] = max(current["fused_score"], item["fused_score"])
            for retriever_name in item["retrievers"]:
                if retriever_name not in current["retrievers"]:
                    current["retrievers"].append(retriever_name)
            current["raw_scores"].update(item["raw_scores"])
        else:
            current["block"]["block_lines"] = _deduplicate_consecutive_lines(current["block"]["block_lines"])
            current["block"].pop("end_index", None)
            current.pop("_signature", None)
            merged_items.append(current)
            current = {
                "block": {
                    "block_id": block["block_id"],
                    "source": block["source"],
                    "block_kind": block["block_kind"],
                    "start_index": block["start_index"],
                    "end_index": block["end_index"],
                    "block_lines": list(block["block_lines"]),
                    "repo_name": block["repo_name"],
                    "file_paths": list(block["file_paths"]),
                },
                "fused_score": item["fused_score"],
                "retrievers": list(item["retrievers"]),
                "raw_scores": dict(item["raw_scores"]),
                "_signature": block_signature,
            }

    current["block"]["block_lines"] = _deduplicate_consecutive_lines(current["block"]["block_lines"])
    current["block"].pop("end_index", None)
    current.pop("_signature", None)
    merged_items.append(current)
    return merged_items


def _summarize_file_paths(file_paths):
    if not file_paths:
        return "unknown"
    if len(file_paths) <= 2:
        return " | ".join(file_paths)
    return f"{file_paths[0]} | {file_paths[1]} | +{len(file_paths) - 2} more"


def _render_selected_items(selected_items):
    output_lines = []
    for rank, item in enumerate(selected_items, start=1):
        block = item["block"]
        raw_scores = item.get("raw_scores") or {}
        absolute_scores = item.get("absolute_scores") or {}
        retrieval_source = block.get("source", "unknown")
        retrieval_repo = block.get("repo") or block.get("repo_name") or "unknown"
        retrieval_file_paths = block.get("file_paths") or []
        if not retrieval_file_paths:
            single_path = block.get("file_path")
            if single_path:
                retrieval_file_paths = [single_path]
        retrieval_retrievers = item.get("retrievers") or []
        if raw_scores:
            raw_score_text = ", ".join(
                f"{name}: {score:.4f}" for name, score in raw_scores.items()
            )
        else:
            raw_score_text = "none"
        if absolute_scores:
            absolute_score_text = ", ".join(
                f"{name}: {score:.4f}" for name, score in absolute_scores.items()
            )
        else:
            absolute_score_text = "none"
        output_lines.append(f"===== block {rank} （raw_scores：{raw_score_text}） =====")
        token_cosine = absolute_scores.get("token_cosine") if absolute_scores else None
        if token_cosine is None:
            relevance_hint_text = "none"
        else:
            relevance_hint_text = (
                f"token_overlap_cosine: {token_cosine:.4f}"
                f" ({_classify_token_cosine(token_cosine)})"
            )
        output_lines.append(f"// Retrieval relevance_hint: {relevance_hint_text}")
        output_lines.append(f"// Retrieval source: {retrieval_source}")
        output_lines.append(f"// Retrieval repo: {retrieval_repo}")
        output_lines.append(
            f"// Retrieval file(s): {_summarize_file_paths(retrieval_file_paths)}"
        )
        output_lines.append(
            "// Retrieval retriever(s): "
            + (" | ".join(retrieval_retrievers) if retrieval_retrievers else "unknown")
        )
        output_lines.extend(block["block_lines"])
        output_lines.append("")
    return output_lines


def search_from_text(
    prompt,
    output_path="result.txt",
    block_order=None,
    merge_overlapping_blocks=None,
    top_k=None,
    retriever_specs=None,
    external_corpus_path=None,
    external_mix_ratio=None,
    external_prepend_blocks=None,
    external_file_top_k=None,
    fusion_k=None,
    query_file_path=None,
    local_blocks=None,
    enable_external_file_coarse_to_fine=None,
    enable_external_composite_file_ranking=None,
    enable_external_filename_match_boost=None,
    enable_external_generic_large_file_penalty=None,
):
    inspection = inspect_retrieval(
        prompt=prompt,
        output_path=output_path,
        block_order=block_order,
        merge_overlapping_blocks=merge_overlapping_blocks,
        top_k=top_k,
        retriever_specs=retriever_specs,
        external_corpus_path=external_corpus_path,
        external_mix_ratio=external_mix_ratio,
        external_prepend_blocks=external_prepend_blocks,
        external_file_top_k=external_file_top_k,
        fusion_k=fusion_k,
        query_file_path=query_file_path,
        local_blocks=local_blocks,
        enable_external_file_coarse_to_fine=enable_external_file_coarse_to_fine,
        enable_external_composite_file_ranking=enable_external_composite_file_ranking,
        enable_external_filename_match_boost=enable_external_filename_match_boost,
        enable_external_generic_large_file_penalty=enable_external_generic_large_file_penalty,
    )
    return inspection["rendered_lines"]


def inspect_retrieval(
    prompt,
    output_path="result.txt",
    block_order=None,
    merge_overlapping_blocks=None,
    top_k=None,
    retriever_specs=None,
    external_corpus_path=None,
    external_mix_ratio=None,
    external_prepend_blocks=None,
    external_file_top_k=None,
    fusion_k=None,
    query_file_path=None,
    local_blocks=None,
    masked_type=None,
    enable_external_file_coarse_to_fine=None,
    enable_external_composite_file_ranking=None,
    enable_external_filename_match_boost=None,
    enable_external_generic_large_file_penalty=None,
):
    # Keep external_mix_ratio in the public signature for backward compatibility.
    # New behavior is controlled by external_prepend_blocks only.
    resolved_block_order = _resolve_block_order(block_order)
    resolved_merge = MERGE_OVERLAPPING_BLOCKS if merge_overlapping_blocks is None else bool(merge_overlapping_blocks)
    resolved_top_k = _resolve_positive_int(TOP_K_BLOCKS if top_k is None else top_k, "top_k")
    resolved_external_prepend = _resolve_positive_int(
        EXTERNAL_PREPEND_BLOCKS if external_prepend_blocks is None else external_prepend_blocks,
        "external_prepend_blocks",
    ) if (EXTERNAL_PREPEND_BLOCKS if external_prepend_blocks is None else external_prepend_blocks) else 0
    resolved_fusion_k = _resolve_positive_int(FUSION_RRF_K if fusion_k is None else fusion_k, "fusion_k")
    resolved_external_file_top_k = _resolve_positive_int(
        EXTERNAL_FILE_TOP_K if external_file_top_k is None else external_file_top_k,
        "external_file_top_k",
    ) if (EXTERNAL_FILE_TOP_K if external_file_top_k is None else external_file_top_k) else 0
    resolved_external_file_coarse_to_fine = (
        ENABLE_EXTERNAL_FILE_COARSE_TO_FINE
        if enable_external_file_coarse_to_fine is None
        else bool(enable_external_file_coarse_to_fine)
    )
    resolved_external_composite_file_ranking = (
        ENABLE_EXTERNAL_COMPOSITE_FILE_RANKING
        if enable_external_composite_file_ranking is None
        else bool(enable_external_composite_file_ranking)
    )
    resolved_external_filename_match_boost = (
        ENABLE_EXTERNAL_FILENAME_MATCH_BOOST
        if enable_external_filename_match_boost is None
        else bool(enable_external_filename_match_boost)
    )
    resolved_external_generic_large_file_penalty = (
        ENABLE_EXTERNAL_GENERIC_LARGE_FILE_PENALTY
        if enable_external_generic_large_file_penalty is None
        else bool(enable_external_generic_large_file_penalty)
    )
    normalized_specs = _normalize_retriever_specs(retriever_specs)
    enabled_local_specs = [spec for spec in normalized_specs if spec["enabled"] and spec["source"] == "local"]
    enabled_external_specs = [spec for spec in normalized_specs if spec["enabled"] and spec["source"] == "external"]

    local_top_k_values = [
        int(spec.get("top_k"))
        for spec in enabled_local_specs
        if spec.get("top_k") is not None
    ]
    external_top_k_values = [
        int(spec.get("top_k"))
        for spec in enabled_external_specs
        if spec.get("top_k") is not None
    ]
    allowed_local_top_k = min(local_top_k_values) if local_top_k_values else resolved_top_k
    allowed_external_prepend = min(external_top_k_values) if external_top_k_values else resolved_external_prepend
    allowed_local_top_k = max(0, allowed_local_top_k)
    allowed_external_prepend = max(0, allowed_external_prepend)

    resolved_local_blocks = local_blocks if local_blocks is not None else _build_local_blocks(prompt, query_file_path=query_file_path)
    if not resolved_local_blocks:
        return []

    sample_lines = get_sample([line.rstrip() for line in str(prompt or "").splitlines() if line.strip()], WINDOW_SIZE)
    sample_tokens = tokenize("\n".join(sample_lines))

    need_external = (
        allowed_external_prepend > 0
        and bool(enabled_external_specs)
    )
    external_cache = _build_external_blocks(external_corpus_path) if need_external else None

    local_ranked_lists = []
    external_ranked_lists = []
    local_target_for_ranking = allowed_local_top_k
    external_target_for_ranking = max(allowed_external_prepend * 4, allowed_external_prepend)

    for spec in normalized_specs:
        spec_top_k = spec.get("top_k")
        if spec["source"] == "local" and local_target_for_ranking <= 0:
            continue
        if spec["source"] == "external" and allowed_external_prepend <= 0:
            continue
        ranked_items = _rank_blocks_for_spec(
            spec=spec,
            local_blocks=resolved_local_blocks,
            external_cache=external_cache,
            sample_tokens=sample_tokens,
            top_k=(
                min(local_target_for_ranking, int(spec_top_k))
                if spec["source"] == "local" and spec_top_k is not None
                else min(allowed_external_prepend, int(spec_top_k))
                if spec["source"] == "external" and spec_top_k is not None
                else local_target_for_ranking if spec["source"] == "local" else external_target_for_ranking
            ),
            external_file_top_k=resolved_external_file_top_k if spec["source"] == "external" else 0,
            query_file_path=query_file_path if spec["source"] == "external" else None,
            query_text=prompt if spec["source"] == "external" else None,
            masked_type=masked_type,
            enable_external_file_coarse_to_fine=resolved_external_file_coarse_to_fine,
            enable_external_composite_file_ranking=resolved_external_composite_file_ranking,
            enable_external_filename_match_boost=resolved_external_filename_match_boost,
            enable_external_generic_large_file_penalty=resolved_external_generic_large_file_penalty,
        )
        if ranked_items:
            bucket = local_ranked_lists if spec["source"] == "local" else external_ranked_lists
            bucket.append((spec["name"], spec["weight"], ranked_items))

    if enabled_local_specs and local_target_for_ranking > 0 and not local_ranked_lists:
        fallback_local = [
            {
                "block": block,
                "fused_score": 1.0,
                "retrievers": ["fallback_local"],
                "raw_scores": {},
            }
            for block in resolved_local_blocks[:local_target_for_ranking]
        ]
        ordered_local_items = _order_selected_items(fallback_local, resolved_block_order)
        _attach_absolute_scores(ordered_local_items, sample_tokens)
        selected_items = ordered_local_items
        if resolved_merge:
            selected_items = _merge_selected_blocks(selected_items)
            _attach_absolute_scores(selected_items, sample_tokens)
        return {
            "rendered_lines": _render_selected_items(selected_items),
            "selected_items": selected_items,
            "ordered_local_items": ordered_local_items,
            "ordered_external_items": [],
            "resolved_block_order": resolved_block_order,
            "resolved_top_k": local_target_for_ranking,
            "resolved_external_prepend": allowed_external_prepend,
            "resolved_external_file_top_k": resolved_external_file_top_k,
            "resolved_fusion_k": resolved_fusion_k,
            "resolved_external_file_coarse_to_fine": resolved_external_file_coarse_to_fine,
            "resolved_external_composite_file_ranking": resolved_external_composite_file_ranking,
            "resolved_external_filename_match_boost": resolved_external_filename_match_boost,
            "resolved_external_generic_large_file_penalty": resolved_external_generic_large_file_penalty,
        }

    fused_local_items = _apply_rrf_fusion(local_ranked_lists, resolved_fusion_k)[:local_target_for_ranking]
    ordered_local_items = _order_selected_items(fused_local_items, resolved_block_order)
    _attach_absolute_scores(ordered_local_items, sample_tokens)

    ordered_external_items = []
    if allowed_external_prepend > 0 and external_ranked_lists:
        fused_external_items = _apply_rrf_fusion(external_ranked_lists, resolved_fusion_k)
        deduped_external_items = _dedupe_external_items(
            fused_external_items,
            max_blocks_per_file=EXTERNAL_MAX_BLOCKS_PER_FILE,
        )
        ordered_external_items = _order_external_items(
            deduped_external_items[:allowed_external_prepend],
        )
        _attach_absolute_scores(ordered_external_items, sample_tokens)

    selected_items = ordered_external_items + ordered_local_items
    if resolved_merge:
        selected_items = _merge_selected_blocks(selected_items)
    _attach_absolute_scores(selected_items, sample_tokens)
    return {
        "rendered_lines": _render_selected_items(selected_items),
        "selected_items": selected_items,
        "ordered_local_items": ordered_local_items,
        "ordered_external_items": ordered_external_items,
        "resolved_block_order": resolved_block_order,
        "resolved_top_k": local_target_for_ranking,
        "resolved_external_prepend": allowed_external_prepend,
        "resolved_external_file_top_k": resolved_external_file_top_k,
        "resolved_fusion_k": resolved_fusion_k,
        "resolved_external_file_coarse_to_fine": resolved_external_file_coarse_to_fine,
        "resolved_external_composite_file_ranking": resolved_external_composite_file_ranking,
        "resolved_external_filename_match_boost": resolved_external_filename_match_boost,
        "resolved_external_generic_large_file_penalty": resolved_external_generic_large_file_penalty,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_txt", type=str, default=None, help="输入 txt 文件路径")
    parser.add_argument("--output_txt", type=str, default="result.txt", help="输出结果文件路径")
    parser.add_argument("--format_verilog", action="store_true", help="仅对输入文件进行 Verilog 格式化处理（只格式化最后一行）")
    args = parser.parse_args()

    if not args.input_txt:
        print("请通过以下方式之一使用：")
        print("1. 使用 --input_txt 参数指定输入文件")
        print("2. 直接导入 search_from_text 函数并传入文本内容")
        return

    if not os.path.exists(args.input_txt):
        print(f"文件不存在：{args.input_txt}")
        return

    if args.format_verilog:
        with open(args.input_txt, "r", encoding="utf-8") as file_obj:
            lines = file_obj.readlines()

        if lines:
            last_line = lines[-1]
            formatted_last_line = format_verilog_line(last_line)
            formatted_last_line = "\n".join(
                [line.strip() for line in formatted_last_line.splitlines() if line.strip()]
            )
            lines[-1] = formatted_last_line + "\n"

            with open(args.input_txt, "w", encoding="utf-8") as file_obj:
                file_obj.writelines(lines)
            print("Verilog 格式化完成！")
        return

    text = "\n".join(read_file(args.input_txt))
    output_lines = search_from_text(text)

    with open(args.output_txt, "w", encoding="utf-8") as file_obj:
        for line in output_lines:
            file_obj.write(line + "\n")
    print(f"检索结果已输出到 {args.output_txt}")


if __name__ == "__main__":
    main()
