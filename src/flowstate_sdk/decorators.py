import asyncio
import functools
import uuid
from typing import Any, Callable, Dict, Optional

from . import context
from .enums import TaskTypes
from .sdk_client import SDKClient
from .task_context import TaskContext


class NoActiveRunError(Exception):
    pass


def sdk_task(
    task_step_id: Optional[str] = None,
    type: TaskTypes = TaskTypes.GENERIC,
    metadata: Optional[Dict[Any, Any]] = None,
    client: Optional[SDKClient] = None,
    inject_context: bool = False,
):

    metadata = metadata or {}

    def decorator(func: Callable):
        is_coro = asyncio.iscoroutinefunction(func)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            nonlocal client
            resolved_client = client
            if not resolved_client:
                curr_run = context.current_run.get()
                if not curr_run:
                    raise NoActiveRunError(
                        "call Workflow.start_run() before using sdk_task."
                    )
                resolved_client = curr_run.client

            resolved_task_id = task_step_id or uuid.uuid4()
            ctx = TaskContext(
                task_step_id=resolved_task_id,
                client=resolved_client,
                func_name=f"{func.__module__}.{func.__qualname__}",
                type=type,
                metadata=metadata,
            )
            ctx.__enter__()

            try:
                if inject_context:
                    result = func(*args, _task_context=ctx, **kwargs)
                else:
                    result = func(*args, **kwargs)
            except Exception:
                import sys

                exc_info = sys.exc_info()
                ctx.__exit__(exc_info[0], exc_info[1], exc_info[2])
                raise
            else:
                ctx.__exit__(None, None, None)
                return result

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            nonlocal client
            resolved_client = client
            if not resolved_client:
                curr_run = context.current_run.get()
                if not curr_run:
                    raise NoActiveRunError(
                        "call Workflow.start_run() before using sdk_task."
                    )
                resolved_client = curr_run.client

            resolved_task_id = task_step_id or uuid.uuid4()
            ctx = TaskContext(
                task_step_id=resolved_task_id,
                client=resolved_client,
                func_name=f"{func.__module__}.{func.__qualname__}",
                type=type,
                metadata=metadata,
            )
            ctx.__enter__()
            try:
                if inject_context:
                    result = await func(*args, _task_context=ctx, **kwargs)
                else:
                    result = await func(*args, **kwargs)
            except Exception:
                import sys

                exc_info = sys.exc_info()
                ctx.__exit__(exc_info[0], exc_info[1], exc_info[2])
                raise
            else:
                ctx.__exit__(None, None, None)
                return result

        sync_wrapper.__globals__.update(func.__globals__)
        async_wrapper.__globals__.update(func.__globals__)

        return async_wrapper if is_coro else sync_wrapper

    return decorator
