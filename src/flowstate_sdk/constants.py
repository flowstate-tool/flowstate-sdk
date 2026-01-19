SDK_ENDPOINT_TIMEOUT: int = 5
# im changing to use the backend's expected auth header name
AUTH_HEADER_NAME: str = "auth"
INVALID_AUTH_MESSAGE: str = "Invalid authentication!"

PROVIDER_NOT_SUPPORTED = "Provider not supported/invalid provider!"
METRICS_DUMP_ERROR = "Error sending metrics to backend!"
NO_TASK_CONTEXT = "No task context available!"

TASK_STEP_ID = "task_step_id"

OPENAI = "openai"
ANTHROPIC = "anthropic"
GEMINI = "gemini"
GOOGLE = "google"

GENERIC_NAME = "generic"
LLM_NAME = "llm"
TOOL_NAME = "tool"
AGENT_NAME = "agent"

USER = "user"

BASE_ENDPOINT = "https://api.tryflowstate.dev"
WORKFLOWS_RESOLVE_ENDPOINT = "https://api.tryflowstate.dev/sdk/metrics"
AUTH_ENDPOINT = "https://api.tryflowstate.dev/sdk/auth"
METRICS_ENDPOINT = "https://api.tryflowstate.dev/sdk/workflows/resolve"

SKIP_SDK_METRICS_FLAG = "SKIP_SDK_METRICS"
