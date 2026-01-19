from typing import Any, Dict, Optional

from .enums import TaskTypes
from .sdk_client import SDKClient
from .task_context import TaskContext


class Run:
    def __init__(self, run_id: str, client: SDKClient) -> None:
        self.run_id: str = run_id
        self.client: SDKClient = client
        self.workflow_id: str | None = None

    def step(
        self,
        task_step_id: str,
        task_type: Optional[TaskTypes],
        metadata: Dict[Any, Any] = {},
    ) -> TaskContext:
        return TaskContext(
            task_step_id=task_step_id,
            client=self.client,
            type=task_type,
            metadata=metadata,
        )

    def end(self) -> None:
        pass
