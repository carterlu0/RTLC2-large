try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

import json
import threading
import time

OPENAI_CLIENT_TIMEOUT_SECONDS = 60.0
OPENAI_CLIENT_MAX_RETRIES = 0


def _model_dump(obj):
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool, list, dict)):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    return obj


def _extract_text_from_sse_payload(text):
    if not isinstance(text, str):
        return ""

    if "event:" not in text or "data:" not in text:
        return ""

    collected_chunks = []
    fallback_text = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or not line.startswith("data:"):
            continue

        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue

        try:
            parsed = json.loads(payload)
        except Exception:
            fallback_text.append(payload)
            continue

        if isinstance(parsed, dict):
            delta = parsed.get("delta")
            if isinstance(delta, str):
                collected_chunks.append(delta)

            text_value = parsed.get("text")
            if isinstance(text_value, str):
                collected_chunks.append(text_value)

            item = parsed.get("item")
            if isinstance(item, dict):
                for content_item in item.get("content", []):
                    if not isinstance(content_item, dict):
                        continue
                    content_text = content_item.get("text")
                    if isinstance(content_text, str):
                        collected_chunks.append(content_text)

    if collected_chunks:
        return "".join(collected_chunks)
    if fallback_text:
        return "\n".join(fallback_text)
    return ""


def _extract_text_from_chat_completion(response):
    if response is None:
        return ""
    if isinstance(response, str):
        return _extract_text_from_sse_payload(response) or response

    choices = getattr(response, "choices", None)
    if choices:
        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        content = getattr(message, "content", None) if message is not None else None
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                item_text = getattr(item, "text", None)
                if isinstance(item_text, str):
                    text_parts.append(item_text)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
            if text_parts:
                return "".join(text_parts)

        text = getattr(first_choice, "text", None)
        if isinstance(text, str):
            return text

    response_dict = _model_dump(response)
    if isinstance(response_dict, dict):
        for choice in response_dict.get("choices", []):
            message = choice.get("message", {})
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                text_parts = [item.get("text", "") for item in content if isinstance(item, dict)]
                if text_parts:
                    return "".join(text_parts)

            text = choice.get("text")
            if isinstance(text, str):
                return text

    return ""


def _extract_text_from_response_api(response):
    if response is None:
        return ""
    if isinstance(response, str):
        return _extract_text_from_sse_payload(response) or response

    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text

    response_dict = _model_dump(response)
    if isinstance(response_dict, dict):
        dict_output_text = response_dict.get("output_text")
        if isinstance(dict_output_text, str):
            return dict_output_text

        text_parts = []
        for output_item in response_dict.get("output", []):
            if not isinstance(output_item, dict):
                continue
            for content_item in output_item.get("content", []):
                if not isinstance(content_item, dict):
                    continue
                content_type = content_item.get("type")
                if content_type in {"output_text", "text"}:
                    text = content_item.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
        if text_parts:
            return "".join(text_parts)

    return ""


def _call_with_optional_fallback(callable_obj, request_kwargs, optional_keys):
    last_exc = None
    optional_keys = [key for key in optional_keys if key in request_kwargs]

    for drop_count in range(len(optional_keys) + 1):
        attempt_kwargs = dict(request_kwargs)
        for key in optional_keys[:drop_count]:
            attempt_kwargs.pop(key, None)

        for retry_index in range(4):
            try:
                return callable_obj(**attempt_kwargs)
            except Exception as exc:
                last_exc = exc
                message = str(exc).lower()
                is_rate_limited = "429" in message or "rate limit" in message or "ratelimit" in message
                if not is_rate_limited or retry_index >= 3:
                    break
                sleep_seconds = min(12.0, 2.0 * (retry_index + 1))
                print(f"Rate limited, retrying in {sleep_seconds:.1f}s...")
                time.sleep(sleep_seconds)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("request failed without a captured exception")


class OpenAICompatibleEngine:
    def __init__(self, api_base, api_key, model_name="deepseek-chat", timeout_seconds=None):
        self.api_base = api_base
        self.api_key = api_key
        self.model_name = model_name
        self.timeout_seconds = float(timeout_seconds or OPENAI_CLIENT_TIMEOUT_SECONDS)
        self.client = None

        if OpenAI is None:
            print("openai 未安装，跳过 FIM API 调用。")
            return

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=self.timeout_seconds,
            max_retries=OPENAI_CLIENT_MAX_RETRIES,
        )

    def generate_fim(self, prompt, suffix, max_tokens=128, temperature=0.1, stop=None):
        if self.client is None:
            return ""

        try:
            request_kwargs = {
                "model": self.model_name,
                "prompt": prompt,
                "suffix": suffix,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if stop:
                request_kwargs["stop"] = stop

            response = _call_with_optional_fallback(
                self.client.completions.create,
                request_kwargs,
                optional_keys=("stop",),
            )
            return response.choices[0].text
        except Exception as e:
            print(f"Error during FIM API call: {e}")
            return ""

    def generate_responses(
        self,
        prompt,
        max_tokens=128,
        temperature=0.1,
        system_prompt=None,
        chat_fallback=True,
        stop=None,
        reasoning_effort=None,
        text_verbosity=None,
    ):
        if self.client is None:
            return "", None

        request_kwargs = {
            "model": self.model_name,
            "input": prompt,
            "max_output_tokens": max_tokens,
        }
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if system_prompt:
            request_kwargs["instructions"] = system_prompt
        if reasoning_effort:
            request_kwargs["reasoning"] = {"effort": reasoning_effort}
        if text_verbosity:
            request_kwargs["text"] = {"verbosity": text_verbosity}

        try:
            response = _call_with_optional_fallback(
                self.client.responses.create,
                request_kwargs,
                optional_keys=("stop", "text", "reasoning"),
            )
            return _extract_text_from_response_api(response), None
        except Exception as responses_exc:
            if not chat_fallback:
                print(f"Error during Responses API call: {responses_exc}")
                return "", None

            try:
                chat_request = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "logprobs": True,
                    "top_logprobs": 5,
                }
                if temperature is not None:
                    chat_request["temperature"] = temperature
                if system_prompt:
                    chat_request["messages"].insert(0, {"role": "system", "content": system_prompt})
                if stop:
                    chat_request["stop"] = stop
                if reasoning_effort:
                    chat_request["reasoning_effort"] = reasoning_effort
                if text_verbosity:
                    chat_request["verbosity"] = text_verbosity

                response = _call_with_optional_fallback(
                    self.client.chat.completions.create,
                    chat_request,
                    optional_keys=("stop", "verbosity", "reasoning_effort", "top_logprobs"),
                )
                first_choice = response.choices[0] if getattr(response, "choices", None) else None
                logprobs = getattr(first_choice, "logprobs", None) if first_choice is not None else None
                return _extract_text_from_chat_completion(response), logprobs
            except Exception as chat_exc:
                print(f"Error during Responses API call: {responses_exc}")
                print(f"Error during chat fallback call: {chat_exc}")
                return "", None

    def generate_chat(
        self,
        prompt,
        max_tokens=128,
        temperature=0.1,
        system_prompt=None,
        stop=None,
        reasoning_effort=None,
        text_verbosity=None,
    ):
        if self.client is None:
            return "", None

        try:
            messages = [{"role": "user", "content": prompt}]
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})

            request_kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stop": stop,
                "logprobs": True,
                "top_logprobs": 5,
            }
            if reasoning_effort:
                request_kwargs["reasoning_effort"] = reasoning_effort
            if text_verbosity:
                request_kwargs["verbosity"] = text_verbosity
            response = _call_with_optional_fallback(
                self.client.chat.completions.create,
                request_kwargs,
                optional_keys=("stop", "verbosity", "reasoning_effort", "top_logprobs"),
            )
            first_choice = response.choices[0] if getattr(response, "choices", None) else None
            logprobs = getattr(first_choice, "logprobs", None) if first_choice is not None else None
            return _extract_text_from_chat_completion(response), logprobs
        except Exception as e:
            print(f"Error during chat API call: {e}")
            return "", None


openai_engines = {}


def _thread_local_engine_key(api_base, api_key, model_name, timeout_seconds=None):
    normalized_timeout = float(timeout_seconds or OPENAI_CLIENT_TIMEOUT_SECONDS)
    return (threading.get_ident(), api_base, api_key, model_name, normalized_timeout)


def run_fim_inference(api_base, api_key, model_name, prompt, suffix, max_tokens=128, temperature=0.1, stop=None, timeout_seconds=None):
    engine_key = _thread_local_engine_key(api_base, api_key, model_name, timeout_seconds)
    fim_engine = openai_engines.get(engine_key)

    if fim_engine is None:
        fim_engine = OpenAICompatibleEngine(api_base, api_key, model_name, timeout_seconds=timeout_seconds)
        openai_engines[engine_key] = fim_engine

    return fim_engine.generate_fim(prompt, suffix, max_tokens=max_tokens, temperature=temperature, stop=stop)


def run_responses_inference(
    api_base,
    api_key,
    model_name,
    prompt,
    max_tokens=128,
    temperature=0.1,
    system_prompt=None,
    chat_fallback=True,
    stop=None,
    reasoning_effort=None,
    text_verbosity=None,
    timeout_seconds=None,
):
    engine_key = _thread_local_engine_key(api_base, api_key, model_name, timeout_seconds)
    engine = openai_engines.get(engine_key)

    if engine is None:
        engine = OpenAICompatibleEngine(api_base, api_key, model_name, timeout_seconds=timeout_seconds)
        openai_engines[engine_key] = engine

    return engine.generate_responses(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
        chat_fallback=chat_fallback,
        stop=stop,
        reasoning_effort=reasoning_effort,
        text_verbosity=text_verbosity,
    )


def run_chat_inference(
    api_base,
    api_key,
    model_name,
    prompt,
    max_tokens=128,
    temperature=0.1,
    system_prompt=None,
    stop=None,
    reasoning_effort=None,
    text_verbosity=None,
    timeout_seconds=None,
):
    engine_key = _thread_local_engine_key(api_base, api_key, model_name, timeout_seconds)
    engine = openai_engines.get(engine_key)

    if engine is None:
        engine = OpenAICompatibleEngine(api_base, api_key, model_name, timeout_seconds=timeout_seconds)
        openai_engines[engine_key] = engine

    return engine.generate_chat(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
        stop=stop,
        reasoning_effort=reasoning_effort,
        text_verbosity=text_verbosity,
    )


def main():
    api_base = "https://api.deepseek.com/beta"
    api_key = "REPLACE_WITH_VALID_KEY"
    model_name = "deepseek-chat"
    prompt = "def fib(a):"
    suffix = "    return fib(a-1) + fib(a-2)"
    result = run_fim_inference(
        api_base=api_base,
        api_key=api_key,
        model_name=model_name,
        prompt=prompt,
        suffix=suffix,
    )
    print("\n生成的代码补全结果:")
    print("=" * 50)
    print(prompt + result + suffix)
    print("=" * 50)


if __name__ == "__main__":
    main()
