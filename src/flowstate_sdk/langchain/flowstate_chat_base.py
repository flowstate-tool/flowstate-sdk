from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from flowstate_sdk.langchain.telemetry import FlowstateCallbackHandler


def _extract_model_name(kwargs: Dict[str, Any]) -> Optional[str]:
    for key in ("model", "model_name", "model_id"):
        val = kwargs.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return None


def _normalize_callbacks(callbacks: Any) -> List[Any]:
    if callbacks is None:
        return []
    if isinstance(callbacks, list):
        return list(callbacks)
    if isinstance(callbacks, tuple):
        return list(callbacks)
    return [callbacks]


def _has_flowstate_callback(callbacks: Iterable[Any]) -> bool:
    for cb in callbacks:
        if isinstance(cb, FlowstateCallbackHandler):
            return True
    return False


def _merge_callbacks(existing: Any, additional: List[Any]) -> List[Any]:
    existing_list = _normalize_callbacks(existing)
    if _has_flowstate_callback(existing_list):
        return existing_list
    return existing_list + list(additional)


class FlowstateChatModelMixin:
    provider: str = ""

    @classmethod
    def _flowstate_build_callbacks(cls, kwargs: Dict[str, Any]) -> List[Any]:
        model = _extract_model_name(kwargs) or "unknown"
        handler = FlowstateCallbackHandler(cls.provider, model)
        return _merge_callbacks(kwargs.pop("callbacks", None), [handler])

    def _flowstate_init_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        kwargs["callbacks"] = self._flowstate_build_callbacks(kwargs)
        return kwargs
