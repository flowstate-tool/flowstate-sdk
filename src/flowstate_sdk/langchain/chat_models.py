from __future__ import annotations

from typing import Any

from flowstate_sdk.langchain.flowstate_chat_base import FlowstateChatModelMixin

try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover - optional dependency
    ChatOpenAI = None

try:
    from langchain_anthropic import ChatAnthropic
except Exception:  # pragma: no cover - optional dependency
    ChatAnthropic = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:  # pragma: no cover - optional dependency
    ChatGoogleGenerativeAI = None


def _missing_dependency(name: str, package: str) -> None:
    raise ImportError(
        f"{name} requires optional dependency '{package}'. "
        f"Install it to use this Flowstate wrapper."
    )


if ChatOpenAI is not None:

    class FlowstateChatOpenAI(FlowstateChatModelMixin, ChatOpenAI):
        provider = "openai"

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs = self._flowstate_init_kwargs(kwargs)
            super().__init__(*args, **kwargs)

else:

    class FlowstateChatOpenAI(FlowstateChatModelMixin):  # type: ignore[misc]
        provider = "openai"

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _missing_dependency("FlowstateChatOpenAI", "langchain-openai")


if ChatAnthropic is not None:

    class FlowstateChatAnthropic(FlowstateChatModelMixin, ChatAnthropic):
        provider = "anthropic"

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs = self._flowstate_init_kwargs(kwargs)
            super().__init__(*args, **kwargs)

else:

    class FlowstateChatAnthropic(FlowstateChatModelMixin):  # type: ignore[misc]
        provider = "anthropic"

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _missing_dependency("FlowstateChatAnthropic", "langchain-anthropic")


if ChatGoogleGenerativeAI is not None:

    class FlowstateChatGoogle(FlowstateChatModelMixin, ChatGoogleGenerativeAI):
        provider = "google"

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs = self._flowstate_init_kwargs(kwargs)
            super().__init__(*args, **kwargs)

else:

    class FlowstateChatGoogle(FlowstateChatModelMixin):  # type: ignore[misc]
        provider = "google"

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _missing_dependency("FlowstateChatGoogle", "langchain-google-genai")


FlowstateChatClaude = FlowstateChatAnthropic

__all__ = [
    "FlowstateChatOpenAI",
    "FlowstateChatAnthropic",
    "FlowstateChatClaude",
    "FlowstateChatGoogle",
]
