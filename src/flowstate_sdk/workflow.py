import uuid
from collections import defaultdict
from datetime import datetime
import os

from . import context
from .sdk_client import SDKClient
from .shared_dataclasses import RunRecord, WorkflowRecord
from .task import Run


class Workflow:

    def __init__(self, workflow_name: str, api_key: str) -> None:
        self.workflow_name: str = workflow_name
        self.client: SDKClient = SDKClient(api_key=api_key)
        resolved_workflow_id = self.client.resolve_workflow_id(self.workflow_name)
        if resolved_workflow_id:
            try:
                self.workflow_id = uuid.UUID(resolved_workflow_id)
            except ValueError:
                self.workflow_id = uuid.uuid4()
        else:
            self.workflow_id = uuid.uuid4()
        self.run_id: uuid.UUID = uuid.uuid4()

    def start_run(self):
        self.run_id = uuid.uuid4()
        run: Run = Run(self.run_id, self.client)
        context.current_run.set(run)
        context.run_stack.set([])
        context.parent_children_mappings.set(defaultdict(set))

        workflow_record = WorkflowRecord(
            workflow_id=str(self.workflow_id),
            workflow_name=self.workflow_name,
            owner_id=self.client._user_id,
        )
        workflow_payload = {
            k: v for k, v in workflow_record.__dict__.items() if v is not None
        }
        self.client.emit(workflow_payload)

        if os.getenv("IS_FLOWSTATE_REPLAY"):
            replay_list = self.client.retrieve_replay_plan(os.getenv("FLOWSTATE_REPLAY_PARENT_ID"))
            context.replay_counter.set(0)
            context.replay_step_list.set(replay_list)
            context.is_replay.set(True)

        run_payload = RunRecord(
            run_id=str(self.run_id),
            workflow_id=str(self.workflow_id),
            status="running",
        ).__dict__
        run_payload.update(
            {
                "start_time": datetime.utcnow().isoformat() + "Z",
                "total_cost_usd": 0.0,
                "total_latency_sec": 0.0,
                "task_count": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
            }
        )
        self.client.emit(run_payload)

        run.workflow_id = str(self.workflow_id)
        return run
