from collections import defaultdict
from contextvars import ContextVar
from typing import Dict, List, Set

current_run: ContextVar[str] = ContextVar("current_run", default=None)
run_stack: ContextVar[List[str]] = ContextVar("run_stack", default=[])
parent_children_mappings: ContextVar[Dict[str, Set[str]]] = ContextVar(
    "parent_children_mappings", default=defaultdict(set)
)
