import time
import uuid
from dataclasses import asdict
from types import TracebackType
from typing import Any, Dict, List, Optional, Set, Type

from . import context
from .enums import TaskTypes
from .sdk_client import SDKClient
from .shared_dataclasses import ProviderMetrics


class TaskContext:
    def __init__(
        self,
        task_step_id: str,
        client: SDKClient,
        func_name: str,
        type: TaskTypes = TaskTypes.GENERIC,
        metadata: Dict[Any, Any] = {},
    ) -> None:
        self.task_step_id: str = task_step_id
        self.client: SDKClient = client
        self.func_name: str = func_name
        self.type: TaskTypes = type
        self.metadata: Dict[Any, Any] = metadata
        self.parent_task_id: Optional[str] = None
        self.metrics: Dict[Any, Any] = {}
        self._start_ts: float | None = None

    def log_llm_usage(self, provider_metrics: ProviderMetrics) -> None:
        pm = asdict(provider_metrics)
        self.metrics.update(pm)
        self.metrics["task_step_id"] = self.task_step_id
        try:
            self.type = TaskTypes.LLM
        except Exception:
            pass

    def __enter__(self):

        run_stack: List["TaskContext"] = context.run_stack.get()
        if run_stack:
            parent_ctx = run_stack[-1]
            self.parent_task_id = parent_ctx.task_step_id if parent_ctx else None
        parent_children_mappings: Dict[str, Set[str]] = (
            context.parent_children_mappings.get()
        )
        parent_key = str(self.parent_task_id) if self.parent_task_id else "None"
        parent_children_mappings[parent_key].add(self.task_step_id)
        context.parent_children_mappings.set(parent_children_mappings)
        run_stack.append(self)
        context.run_stack.set(run_stack)

        self._start_ts = time.time()
        self.metrics["start_ts"] = self._start_ts

        return self

    # TODO: delete later, deprecated
    def log_tool(
        self,
        tool_name: str,
        metadata: Optional[Dict[Any, Any]] = None,
        latency_sec: Optional[float] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        cost_usd: Optional[float] = None,
        raw_output: Optional[Any] = None,
        tool_type: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> str:

        child_id = task_id or str(uuid.uuid4())

        parent_children_mappings: Dict[str, Set[str]] = (
            context.parent_children_mappings.get()
        )
        parent_key = str(self.task_step_id) if self.task_step_id else "None"
        parent_children_mappings[parent_key].add(child_id)
        context.parent_children_mappings.set(parent_children_mappings)

        curr_run = context.current_run.get()
        run_id = getattr(curr_run, "run_id", None)
        workflow_id = getattr(curr_run, "workflow_id", None)

        payload: Dict[str, Any] = {
            "task_step_id": child_id,
            "parent_task_id": self.task_step_id,
            "children_tasks": [],
            "run_id": str(run_id) if run_id else None,
            "workflow_id": str(workflow_id) if workflow_id else None,
            "type": (tool_type or "tool"),
            "tool_name": tool_name,
        }

        payload_metadata = {"tool_name": tool_name}
        if metadata:
            payload_metadata.update(metadata)
        payload["metadata"] = payload_metadata

        if latency_sec is not None:
            payload["latency_sec"] = latency_sec
        if input_tokens is not None:
            payload["input_tokens"] = input_tokens
        if output_tokens is not None:
            payload["output_tokens"] = output_tokens
        if cost_usd is not None:
            payload["cost_usd"] = cost_usd
        if raw_output is not None:
            try:
                payload["raw_output"] = str(raw_output)
            except Exception:
                payload["raw_output"] = "<unserializable>"

        self.client.emit(payload, task_step_id=child_id)
        return child_id

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        run_stack: List["TaskContext"] = context.run_stack.get()
        if run_stack:
            run_stack.pop()
            context.run_stack.set(run_stack)
        if not exc_type:
            parent_children_mappings = context.parent_children_mappings.get()
            children = list(parent_children_mappings.get(self.task_step_id, []))
            curr_run = context.current_run.get()
            run_id = getattr(curr_run, "run_id", None)
            workflow_id = getattr(curr_run, "workflow_id", None)

            latency = None
            if self._start_ts:
                latency = time.time() - self._start_ts

            payload = dict(self.metrics)
            payload.update(
                {
                    "task_step_id": self.task_step_id,
                    "parent_task_id": self.parent_task_id,
                    "children_tasks": children,
                    "run_id": str(run_id) if run_id else None,
                    "workflow_id": str(workflow_id) if workflow_id else None,
                    "type": (
                        self.type.value
                        if hasattr(self.type, "value")
                        else (
                            self.type.name
                            if hasattr(self.type, "name")
                            else str(self.type)
                        )
                    ).lower(),
                }
            )

            payload.update({"tool_name": self.func_name})

            if latency is not None:
                payload["latency_sec"] = latency

            if self.metadata:
                payload["metadata"] = self.metadata

            self.client.emit(payload, task_step_id=self.task_step_id)
