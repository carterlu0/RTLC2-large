import os
from pathlib import Path

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

try:
    import tiktoken
except ImportError:
    tiktoken = None

try:
    from google.genai.local_tokenizer import LocalTokenizer
except ImportError:
    LocalTokenizer = None

try:
    import anthropic
except ImportError:
    anthropic = None


PLACEHOLDER_MARKERS = (
    "REPLACE_",
    "YOUR_",
    "DUMMY",
    "<",
)


class BaseTokenCounter:
    kind = "base"
    exact = True

    def max_context_tokens(self):
        return None

    def count_text(self, text):
        raise NotImplementedError

    def summary(self):
        return {
            "kind": self.kind,
            "exact": self.exact,
        }


class NullTokenCounter(BaseTokenCounter):
    kind = "disabled"
    exact = False

    def __init__(self, reason):
        self.reason = reason

    def count_text(self, text):
        return None

    def summary(self):
        summary = super().summary()
        summary["reason"] = self.reason
        return summary


class HuggingFaceTokenCounter(BaseTokenCounter):
    kind = "huggingface"

    def __init__(self, tokenizer, tokenizer_path, model_name, max_context_tokens=None):
        self.tokenizer = tokenizer
        self.tokenizer_path = str(tokenizer_path)
        self.model_name = model_name
        self._max_context_tokens = max_context_tokens

    def count_text(self, text):
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            max_length=None,
            return_attention_mask=False,
            return_token_type_ids=False,
            verbose=False,
        )
        return len(encoded["input_ids"])

    def max_context_tokens(self):
        return self._max_context_tokens

    def summary(self):
        summary = super().summary()
        summary.update(
            {
                "model_name": self.model_name,
                "tokenizer_path": self.tokenizer_path,
                "max_context_tokens": self.max_context_tokens(),
            }
        )
        return summary


class TikTokenCounter(BaseTokenCounter):
    kind = "tiktoken"

    def __init__(self, model_name, encoding_name, exact=True, note=None, max_context_tokens=None):
        self.model_name = model_name
        self.encoding_name = encoding_name
        self.exact = exact
        self.note = note
        self._max_context_tokens = max_context_tokens
        self.encoding = tiktoken.get_encoding(encoding_name)

    def count_text(self, text):
        return len(self.encoding.encode(text))

    def max_context_tokens(self):
        return self._max_context_tokens

    def summary(self):
        summary = super().summary()
        summary.update(
            {
                "model_name": self.model_name,
                "encoding_name": self.encoding_name,
                "max_context_tokens": self.max_context_tokens(),
            }
        )
        if self.note:
            summary["note"] = self.note
        return summary


class GeminiLocalTokenCounter(BaseTokenCounter):
    kind = "gemini_local"

    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = LocalTokenizer(model_name=model_name)

    def count_text(self, text):
        return self.tokenizer.count_tokens(text).total_tokens

    def summary(self):
        summary = super().summary()
        summary["model_name"] = self.model_name
        return summary


class AnthropicApiTokenCounter(BaseTokenCounter):
    kind = "anthropic_api"

    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=api_key)
        self.cache = {}

    def count_text(self, text):
        if text in self.cache:
            return self.cache[text]

        result = self.client.messages.count_tokens(
            model=self.model_name,
            messages=[{"role": "user", "content": text}],
        )
        self.cache[text] = result.input_tokens
        return result.input_tokens

    def summary(self):
        summary = super().summary()
        summary["model_name"] = self.model_name
        return summary


def looks_like_placeholder(value):
    if not value:
        return True

    value = str(value).strip()
    if not value:
        return True

    upper_value = value.upper()
    return any(marker in upper_value for marker in PLACEHOLDER_MARKERS)


def infer_model_family(model_name):
    lower_name = str(model_name or "").lower()

    if lower_name.startswith(("gpt-", "chatgpt-", "o1", "o3", "o4")):
        return "openai"
    if "gemini" in lower_name:
        return "gemini"
    if "claude" in lower_name:
        return "claude"
    if any(token in lower_name for token in ("deepseek", "llama", "coder", "qwen", "mistral")):
        return "huggingface"
    return "unknown"


def resolve_huggingface_tokenizer_limit(tokenizer):
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    init_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}

    for candidate in (
        tokenizer_limit,
        init_kwargs.get("model_max_length"),
        init_kwargs.get("max_length"),
    ):
        if isinstance(candidate, int) and 0 < candidate < int(1e9):
            return candidate

    return None


def build_huggingface_counter(model_name, tokenizer_path, max_context_tokens=None):
    if AutoTokenizer is None:
        return NullTokenCounter("transformers 未安装，无法加载 HuggingFace tokenizer。")

    if not tokenizer_path:
        return NullTokenCounter("未提供 HuggingFace tokenizer 路径。")

    tokenizer_path = Path(tokenizer_path)
    if not tokenizer_path.exists():
        return NullTokenCounter(f"HuggingFace tokenizer 路径不存在: {tokenizer_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    except Exception as exc:
        return NullTokenCounter(f"加载 HuggingFace tokenizer 失败: {exc}")

    resolved_limit = resolve_huggingface_tokenizer_limit(tokenizer)

    config_path = tokenizer_path / "config.json"
    if config_path.exists():
        try:
            import json

            config_data = json.loads(config_path.read_text(encoding="utf-8"))
            model_limit = config_data.get("max_position_embeddings")
            if isinstance(model_limit, int) and model_limit > 0:
                resolved_limit = model_limit
        except Exception:
            pass

    if isinstance(max_context_tokens, int) and max_context_tokens > 0:
        resolved_limit = max_context_tokens

    if isinstance(resolved_limit, int) and resolved_limit > 0:
        tokenizer.model_max_length = resolved_limit
        tokenizer.init_kwargs["model_max_length"] = resolved_limit
        tokenizer.init_kwargs["max_length"] = resolved_limit

    return HuggingFaceTokenCounter(
        tokenizer,
        tokenizer_path,
        model_name,
        max_context_tokens=resolved_limit,
    )


def build_openai_counter(model_name, encoding_name=None, max_context_tokens=None):
    if tiktoken is None:
        return NullTokenCounter("tiktoken 未安装，无法加载 OpenAI tokenizer。")

    try:
        resolved_encoding = tiktoken.encoding_for_model(model_name).name
    except Exception:
        resolved_encoding = encoding_name or "o200k_base"

    return TikTokenCounter(
        model_name=model_name,
        encoding_name=resolved_encoding,
        exact=True,
        max_context_tokens=max_context_tokens,
    )


def build_gemini_counter(model_name):
    if LocalTokenizer is None:
        return NullTokenCounter("google-genai 本地 tokenizer 依赖缺失，无法加载 Gemini tokenizer。")

    try:
        return GeminiLocalTokenCounter(model_name)
    except Exception as exc:
        return NullTokenCounter(f"加载 Gemini 本地 tokenizer 失败: {exc}")


def build_claude_counter(model_name, api_key, fallback_encoding="cl100k_base"):
    if anthropic is not None and not looks_like_placeholder(api_key):
        try:
            return AnthropicApiTokenCounter(model_name=model_name, api_key=api_key)
        except Exception as exc:
            note = f"Anthropic 官方 token 计数器初始化失败，改用近似 tiktoken: {exc}"
        else:
            note = None
    else:
        note = "Anthropic 官方本地 tokenizer 不可用，且未提供有效 API key，改用近似 tiktoken。"

    if tiktoken is None:
        return NullTokenCounter("anthropic 可用但 tiktoken 未安装，无法提供 Claude token 近似计数。")

    return TikTokenCounter(
        model_name=model_name,
        encoding_name=fallback_encoding,
        exact=False,
        note=note,
    )


def load_backend_token_counter(backend_name, backend_config, default_local_model_path):
    token_counter_config = backend_config.get("token_counter", {})
    model_name = backend_config.get("model_name", backend_name)
    family = token_counter_config.get("family") or infer_model_family(model_name)
    kind = token_counter_config.get("kind", "auto")
    tokenizer_path = token_counter_config.get("tokenizer_path") or default_local_model_path
    fallback_encoding = token_counter_config.get("fallback_encoding", "cl100k_base")
    max_context_tokens = token_counter_config.get("max_context_tokens")

    if kind == "disabled":
        return NullTokenCounter("配置显式禁用了 token 计数器。")
    if kind in {"huggingface", "hf_local"}:
        return build_huggingface_counter(
            model_name=model_name,
            tokenizer_path=tokenizer_path,
            max_context_tokens=max_context_tokens,
        )
    if kind == "openai_tiktoken":
        return build_openai_counter(
            model_name=model_name,
            encoding_name=token_counter_config.get("encoding_name"),
            max_context_tokens=max_context_tokens,
        )
    if kind == "gemini_local":
        return build_gemini_counter(model_name=model_name)
    if kind == "anthropic_api":
        return build_claude_counter(
            model_name=model_name,
            api_key=backend_config.get("api_key") or os.getenv("ANTHROPIC_API_KEY"),
            fallback_encoding=fallback_encoding,
        )

    if family == "openai":
        return build_openai_counter(
            model_name=model_name,
            encoding_name=token_counter_config.get("encoding_name"),
            max_context_tokens=max_context_tokens,
        )
    if family == "gemini":
        return build_gemini_counter(model_name=model_name)
    if family == "claude":
        return build_claude_counter(
            model_name=model_name,
            api_key=backend_config.get("api_key") or os.getenv("ANTHROPIC_API_KEY"),
            fallback_encoding=fallback_encoding,
        )
    if family == "huggingface":
        return build_huggingface_counter(
            model_name=model_name,
            tokenizer_path=tokenizer_path,
            max_context_tokens=max_context_tokens,
        )

    if tokenizer_path:
        return build_huggingface_counter(
            model_name=model_name,
            tokenizer_path=tokenizer_path,
            max_context_tokens=max_context_tokens,
        )

    return NullTokenCounter(f"无法为 backend {backend_name} 的模型 {model_name} 自动推断 token 计数器。")


def load_token_counters(mode_config, default_local_model_path):
    counters = {}
    for backend_name, backend_config in mode_config.get("backends", {}).items():
        counters[backend_name] = load_backend_token_counter(
            backend_name=backend_name,
            backend_config=backend_config,
            default_local_model_path=default_local_model_path,
        )
    return counters


def summarize_token_counters(token_counters):
    return {
        backend_name: counter.summary()
        for backend_name, counter in token_counters.items()
    }
