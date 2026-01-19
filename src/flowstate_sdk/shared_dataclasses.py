from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ProviderMetrics:
    run_id: str
    provider: str
    model: str
    input_chars: int
    output_chars: int
    latency_sec: float
    tool_name: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    raw_input: Optional[str] = None
    raw_response: Any = field(default=None)


# these two are for workflow and run records from sdk


@dataclass
class WorkflowRecord:
    workflow_id: str
    workflow_name: str
    owner_id: Optional[str] = None


@dataclass
class RunRecord:
    run_id: str
    workflow_id: str
    status: str = "running"
