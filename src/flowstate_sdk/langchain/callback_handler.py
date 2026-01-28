import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from flowstate_sdk import context
from flowstate_sdk.cost_table import COST_TABLE
from flowstate_sdk.enums import TaskTypes
from flowstate_sdk.shared_dataclasses import ProviderMetrics
from flowstate_sdk.task_context import TaskContext
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        chunks: List[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    chunks.append(part["text"])
                else:
                    chunks.append(str(part))
            else:
                chunks.append(str(part))
        return "\n".join(chunks)

    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        return str(content)

    return str(content)


def _extract_system_and_user_from_messages(
    msgs: Optional[List[Any]],
) -> Tuple[Optional[str], Optional[str]]:
    if not msgs:
        return None, None

    system_parts: List[str] = []
    user_parts: List[str] = []

    for m in msgs:
        m_type = getattr(m, "type", None)
        content = getattr(m, "content", None)
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


def _get_messages_from_kwargs(kwargs: Dict[str, Any]) -> Optional[List[Any]]:
    msgs = kwargs.get("messages")
    if msgs is not None:
        return msgs

    inputs = kwargs.get("inputs")
    if isinstance(inputs, dict) and inputs.get("messages") is not None:
        return inputs.get("messages")

    inp = kwargs.get("input")
    if isinstance(inp, dict) and inp.get("messages") is not None:
        return inp.get("messages")

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

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        self._start_ts = time.time()
        self._tool_name = serialized.get("name")

        self._input_str = prompts[0] if prompts else None
        self._input_chars = len(self._input_str) if self._input_str else None

        msgs = _get_messages_from_kwargs(kwargs)
        sys_p, user_p = _extract_system_and_user_from_messages(msgs)

        if (not sys_p) and isinstance(kwargs.get("system"), str):
            sys_p = kwargs["system"].strip() or None
        if (not sys_p) and isinstance(kwargs.get("system_prompt"), str):
            sys_p = kwargs["system_prompt"].strip() or None

        self._system_prompt = sys_p
        self._user_prompt = user_p

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        latency = (time.time() - self._start_ts) if self._start_ts else None

        generations = response.generations
        generation_chunk = None
        if len(generations) > 0 and len(generations[0]) > 0:
            generation_chunk = generations[0][0]
        if not generation_chunk:
            return
        if (
            not generation_chunk.generation_info.get("finish_reason")
            or generation_chunk.generation_info["finish_reason"] != "stop"
        ):
            return

        ai_message_chunk = generation_chunk.message

        usage: Dict[str, Any] = {}
        if ai_message_chunk and getattr(ai_message_chunk, "usage_metadata", None):
            usage = ai_message_chunk.usage_metadata or {}

        input_tokens = usage.get("input_tokens", 0) or 0
        output_tokens = usage.get("output_tokens", 0) or 0

        input_cost_per_token_usd = (
            COST_TABLE.get(self.provider, {}).get(self.model, {}).get("input", 0.0)
        )
        output_cost_per_token_usd = (
            COST_TABLE.get(self.provider, {}).get(self.model, {}).get("output", 0.0)
        )
        cost_usd = (
            input_cost_per_token_usd * input_tokens
            + output_cost_per_token_usd * output_tokens
        )

        output_text = ""
        if response.generations and response.generations[0]:
            output_text = response.generations[0][0].text or ""

        provider_metrics = ProviderMetrics(
            run_id=context.current_run.get(),
            provider=self.provider,
            model=self.model,
            input_chars=self._input_chars,
            output_chars=len(output_text) if output_text else None,
            latency_sec=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_name=self._tool_name,
            cost_usd=cost_usd if cost_usd != 0.0 else None,
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
            metadata={"provider": self.provider, "model": self.model},
        )
        llm_ctx.__enter__()
        try:
            llm_ctx.log_llm_usage(provider_metrics)
        finally:
            llm_ctx.__exit__(None, None, None)
