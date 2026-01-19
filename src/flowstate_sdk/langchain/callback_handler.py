# flowstate_langchain_callback.py
import time
from typing import Any, Dict, List, Optional

from flowstate_sdk import context
from flowstate_sdk.cost_table import COST_TABLE
from flowstate_sdk.shared_dataclasses import ProviderMetrics
from flowstate_sdk.task_context import TaskContext
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class IrisCallbackHandler(BaseCallbackHandler):
    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model
        self._start_ts: Optional[float] = None
        self._input_chars: Optional[int] = None
        self._input_str: Optional[str] = None
        self._tool_name: Optional[str] = None

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
        self._input_chars = len(prompts[0]) if len(prompts) > 0 else None
        self._input = prompts[0] if len(prompts) > 0 else None

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        latency = (time.time() - self._start_ts) if self._start_ts else None

        generations = response.generations
        generation_chunk = None
        if len(generations) > 0 and len(generations[0]) > 0:
            generation_chunk = generations[0][0]
        if not generation_chunk:
            return
        if (
            not generation_chunk.generation_info["finish_reason"]
            or generation_chunk.generation_info["finish_reason"] != "stop"
        ):
            return

        ai_message_chunk = generation_chunk.message

        if ai_message_chunk:
            usage = ai_message_chunk.usage_metadata

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        cost_usd = 0.0
        input_cost_per_token_usd = (
            COST_TABLE.get(self.provider, {}).get(self.model, {}).get("input", 0.0)
        )
        output_cost_per_token_usd = (
            COST_TABLE.get(self.provider, {}).get(self.model, {}).get("output", 0.0)
        )
        cost_usd += (
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
            raw_input=self._input_str,
            raw_response=response,
        )

        task_ctx = self._get_active_task()
        task_ctx.log_llm_usage(provider_metrics)
