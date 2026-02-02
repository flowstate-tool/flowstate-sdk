import os
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

from flowstate_sdk import context
from flowstate_sdk.cost_table import COST_TABLE
from flowstate_sdk.enums import TaskTypes
from flowstate_sdk.shared_dataclasses import ProviderMetrics
from flowstate_sdk.task_context import TaskContext
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage


def _get_messages_from_kwargs(kwargs: Dict[str, Any]) -> Optional[List[Any]]:
    msgs = _unwrap_messages(kwargs.get("messages"))
    if msgs:
        return msgs

    inv = kwargs.get("invocation_params")
    if isinstance(inv, dict):
        msgs = _unwrap_messages(inv.get("messages"))
        if msgs:
            return msgs

    inputs = kwargs.get("inputs")
    if isinstance(inputs, dict):
        msgs = _unwrap_messages(inputs.get("messages"))
        if msgs:
            return msgs

    inp = kwargs.get("input")
    if isinstance(inp, dict):
        msgs = _unwrap_messages(inp.get("messages"))
        if msgs:
            return msgs

    return None


def _unwrap_messages(messages: Any) -> Optional[List[Any]]:
    if messages is None:
        return None
    if isinstance(messages, (list, tuple)):
        return list(messages)
    if hasattr(messages, "messages"):
        try:
            inner = getattr(messages, "messages")
        except Exception:
            inner = None
        if isinstance(inner, (list, tuple)):
            return list(inner)
    return None


def _normalize_messages(messages: Any) -> List[Any]:
    """
    LangChain may pass:
      - List[List[BaseMessage]] (batched)
      - List[BaseMessage]
      - List[dict] (OpenAI-style: {"role": "...", "content": ...})
    This normalizes to a single flat List[Any] of messages for the first batch item.
    """
    if not messages:
        return []
    if isinstance(messages, (list, tuple)):
        # If it's a batch: [[msg1, msg2, ...], [msg1, ...]]
        if messages and isinstance(messages[0], (list, tuple)):
            return list(messages[0])
        return list(messages)
    if hasattr(messages, "messages"):
        inner = _unwrap_messages(messages)
        if inner:
            if inner and isinstance(inner[0], (list, tuple)):
                return list(inner[0])
            return list(inner)
    return []


def _content_to_text(content: Any) -> str:
    """
    Convert message content into text.
    Supports:
      - str
      - list parts (OpenAI multimodal / tool-like parts)
      - dict with "text"
      - anything else -> str(...)
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: List[str] = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    out.append(part["text"])
                else:
                    out.append(str(part))
            else:
                out.append(str(part))
        return "\n".join(out)
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        return str(content)
    return str(content)


def _msg_type_and_content(m: Any) -> Tuple[Optional[str], Any]:
    """
    Extract (type/role, content) from either:
      - LangChain BaseMessage objects (m.type, m.content)
      - OpenAI-style dicts ({'role': 'user'|'system'|'assistant', 'content': ...})
    Normalize role names to LangChain-like:
      user -> human
      assistant -> ai
    """

    if isinstance(m, SystemMessage):
        return "system", m.content
    if isinstance(m, HumanMessage):
        return "human", m.content

    # LangChain message objects
    if hasattr(m, "type") or hasattr(m, "content"):
        msg_type = getattr(m, "type", None)
        content = getattr(m, "content", None)
        role = getattr(m, "role", None) if hasattr(m, "role") else None
        if msg_type in (None, "chat") and role:
            msg_type = role
        if msg_type == "user":
            msg_type = "human"
        elif msg_type == "assistant":
            msg_type = "ai"
        return msg_type, content

    # Dict messages (OpenAI-style)
    if isinstance(m, dict):
        role = m.get("role") or m.get("type")
        content = m.get("content")
        if role == "user":
            role = "human"
        elif role == "assistant":
            role = "ai"
        return role, content

    return None, None


def _extract_system_and_user_from_messages(
    msgs: Optional[List[Any]],
) -> Tuple[Optional[str], Optional[str]]:
    if not msgs:
        return None, None

    system_parts: List[str] = []
    user_parts: List[str] = []

    for m in msgs:
        m_type, content = _msg_type_and_content(m)
        text = _content_to_text(content).strip()
        if not text:
            continue

        if m_type == "system":
            system_parts.append(text)
        elif m_type == "human":
            user_parts.append(text)

    system_prompt = "\n\n".join(system_parts).strip() or None
    user_prompt = "\n\n".join(user_parts).strip() or None
    return system_prompt, user_prompt


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str) and value.strip() != "":
            return int(float(value))
    except Exception:
        return None
    return None


def _iter_usage_dicts(candidates: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for data in candidates:
        if not isinstance(data, dict):
            continue
        yield data
        for key in ("token_usage", "usage", "usage_metadata"):
            nested = data.get(key)
            if isinstance(nested, dict):
                yield nested


def _extract_token_usage(candidates: Iterable[Dict[str, Any]]) -> Tuple[Optional[int], Optional[int]]:
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    for usage in _iter_usage_dicts(candidates):
        if input_tokens is None:
            input_tokens = _coerce_int(usage.get("input_tokens"))
            if input_tokens is None:
                input_tokens = _coerce_int(usage.get("prompt_tokens"))
        if output_tokens is None:
            output_tokens = _coerce_int(usage.get("output_tokens"))
            if output_tokens is None:
                output_tokens = _coerce_int(usage.get("completion_tokens"))
        if input_tokens is not None and output_tokens is not None:
            break
    return input_tokens, output_tokens


def _normalize_model_name(model: Optional[str]) -> Optional[str]:
    if not model or not isinstance(model, str):
        return model
    model = model.strip()
    if model.startswith("gpt") and not model.startswith("gpt-"):
        if len(model) > 3 and model[3].isdigit():
            model = "gpt-" + model[3:]
    return model


def _is_tool_call_message(message: Any, gen_info: Optional[Dict[str, Any]]) -> bool:
    if isinstance(gen_info, dict):
        finish_reason = gen_info.get("finish_reason")
        if finish_reason in ("tool_calls", "function_call"):
            return True

    if message is None:
        return False

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        return True

    additional_kwargs = getattr(message, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict):
        if additional_kwargs.get("tool_calls") or additional_kwargs.get("function_call"):
            return True

    return False


def _estimate_token_count(text: Optional[str], model: Optional[str]) -> Optional[int]:
    if not text:
        return 0
    try:
        import tiktoken  # type: ignore
    except Exception:
        tiktoken = None

    if tiktoken is not None:
        try:
            try:
                enc = tiktoken.encoding_for_model(model or "")
            except Exception:
                enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            return None

    if os.getenv("FLOWSTATE_APPROX_TOKENS"):
        approx = (len(text) + 3) // 4
        return approx if approx > 0 else 0

    return None


class FlowstateCallbackHandler(BaseCallbackHandler):
    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model
        self._start_ts: Optional[float] = None
        self._input_chars: Optional[int] = None
        self._input_str: Optional[str] = None
        self._tool_name: Optional[str] = None
        self._system_prompt: Optional[str] = None
        self._user_prompt: Optional[str] = None

    def _get_active_task(self) -> TaskContext:
        run_stack: List[TaskContext] = context.run_stack.get()
        if not run_stack:
            raise RuntimeError(
                "No active TaskContext. Wrap your step with @sdk_task or run.step()."
            )
        return run_stack[-1]

    # ---- Chat model callbacks (preferred for ChatOpenAI, etc.) ----

    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[Any]], **kwargs: Any
    ) -> None:
        self._start_ts = time.time()
        self._tool_name = serialized.get("name")

        msgs = _normalize_messages(messages)
        if not msgs:
            msgs = _get_messages_from_kwargs(kwargs) or []

        self._system_prompt, self._user_prompt = _extract_system_and_user_from_messages(
            msgs
        )

        # Build a raw input string by concatenating message contents.
        input_parts: List[str] = []
        for m in msgs:
            _, content = _msg_type_and_content(m)
            text = _content_to_text(content).strip()
            if text:
                input_parts.append(text)

        self._input_str = "\n\n".join(input_parts).strip() or None
        self._input_chars = len(self._input_str) if self._input_str else None

    def on_chat_model_end(self, response: LLMResult, **kwargs: Any) -> None:
        self._log_from_llm_result(response)

    # ---- LLM callbacks (for legacy/non-chat or alternate code paths) ----

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        self._start_ts = time.time()
        self._tool_name = serialized.get("name")

        self._input_str = prompts[0] if prompts else None
        self._input_chars = len(self._input_str) if self._input_str else None

        msgs = _get_messages_from_kwargs(kwargs)
        self._system_prompt, self._user_prompt = _extract_system_and_user_from_messages(
            msgs
        )

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self._log_from_llm_result(response)

    # ---- Shared logging logic ----

    def _log_from_llm_result(self, response: LLMResult) -> None:
        latency = (time.time() - self._start_ts) if self._start_ts else None

        generations = response.generations
        generation_chunk = None
        if len(generations) > 0 and len(generations[0]) > 0:
            generation_chunk = generations[0][0]
        if not generation_chunk:
            return

        gen_info = generation_chunk.generation_info or {}
        finish_reason = gen_info.get("finish_reason")
        ai_message_chunk = getattr(generation_chunk, "message", None)

        if finish_reason and finish_reason != "stop":
            return
        if finish_reason is None and _is_tool_call_message(ai_message_chunk, gen_info):
            return

        llm_output = getattr(response, "llm_output", None)
        resolved_model = self.model
        if isinstance(llm_output, dict):
            for key in ("model_name", "model", "model_id"):
                val = llm_output.get(key)
                if isinstance(val, str) and val.strip():
                    resolved_model = val
                    break
        resolved_model = _normalize_model_name(resolved_model)

        # Extract output text robustly (chat message content or plain .text)
        output_text = ""
        if response.generations and response.generations[0]:
            generation = response.generations[0][0]
            if getattr(generation, "message", None) is not None:
                output_text = _content_to_text(generation.message.content)
            else:
                output_text = generation.text or ""

        usage_candidates: List[Dict[str, Any]] = []
        gen_response_metadata = getattr(generation_chunk, "response_metadata", None)
        if isinstance(gen_response_metadata, dict):
            usage_candidates.append(gen_response_metadata)
        gen_additional_kwargs = getattr(generation_chunk, "additional_kwargs", None)
        if isinstance(gen_additional_kwargs, dict):
            usage_candidates.append(gen_additional_kwargs)
        if ai_message_chunk is not None:
            usage_metadata = getattr(ai_message_chunk, "usage_metadata", None)
            if isinstance(usage_metadata, dict):
                usage_candidates.append(usage_metadata)
            response_metadata = getattr(ai_message_chunk, "response_metadata", None)
            if isinstance(response_metadata, dict):
                usage_candidates.append(response_metadata)
            additional_kwargs = getattr(ai_message_chunk, "additional_kwargs", None)
            if isinstance(additional_kwargs, dict):
                usage_candidates.append(additional_kwargs)
        if isinstance(gen_info, dict) and gen_info:
            usage_candidates.append(gen_info)
        if isinstance(llm_output, dict):
            usage_candidates.append(llm_output)

        input_tokens, output_tokens = _extract_token_usage(usage_candidates)
        if input_tokens is None:
            input_tokens = _estimate_token_count(self._input_str, resolved_model)
        if output_tokens is None:
            output_tokens = _estimate_token_count(output_text, resolved_model)

        if os.getenv("FLOWSTATE_DEBUG_LLM_USAGE"):
            try:
                import json

                debug_payload = {
                    "finish_reason": finish_reason,
                    "usage_candidates": usage_candidates,
                    "llm_output": llm_output,
                    "resolved_model": resolved_model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
                print(
                    "FlowstateCallbackHandler.usage_debug =>",
                    json.dumps(debug_payload, default=str)[:4000],
                )
            except Exception:
                print("FlowstateCallbackHandler.usage_debug => (failed to serialize)")

        input_cost_per_token_usd = (
            COST_TABLE.get(self.provider, {}).get(resolved_model, {}).get("input", 0.0)
        )
        output_cost_per_token_usd = (
            COST_TABLE.get(self.provider, {}).get(resolved_model, {}).get("output", 0.0)
        )
        cost_usd = None
        if input_tokens is not None or output_tokens is not None:
            input_token_count = input_tokens or 0
            output_token_count = output_tokens or 0
            cost_value = (
                input_cost_per_token_usd * input_token_count
                + output_cost_per_token_usd * output_token_count
            )
            if cost_value != 0.0:
                cost_usd = cost_value

        provider_metrics = ProviderMetrics(
            run_id=context.current_run.get(),
            provider=self.provider,
            model=resolved_model,
            input_chars=self._input_chars,
            output_chars=len(output_text) if output_text else None,
            latency_sec=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_name=self._tool_name,
            cost_usd=cost_usd,
            raw_input=self._input_str,
            system_prompt=self._system_prompt,
            user_prompt=self._user_prompt,
            raw_response=response,
        )

        task_ctx = self._get_active_task()
        llm_ctx = TaskContext(
            task_step_id=str(uuid.uuid4()),
            client=task_ctx.client,
            func_name=f"{task_ctx.func_name}.llm",
            type=TaskTypes.LLM,
            metadata={"provider": self.provider, "model": resolved_model},
        )
        llm_ctx.__enter__()
        try:
            llm_ctx.log_llm_usage(provider_metrics)
        finally:
            llm_ctx.__exit__(None, None, None)
