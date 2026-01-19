from enum import Enum

from . import constants


class TaskTypes(Enum):
    GENERIC = constants.GENERIC_NAME
    LLM = constants.LLM_NAME
    TOOL = constants.TOOL_NAME
    AGENT = constants.AGENT_NAME
