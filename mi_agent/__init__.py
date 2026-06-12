"""MI Agent v1 foundation.

An isolated, additive package that provides the building blocks for a
Management Information (MI) querying agent.  It does NOT touch the existing
ESMA / onboarding / validation / reporting pipeline.

Public surface:
  - MIQuerySpec               : v1 query specification (mi_query_spec)
  - validate_mi_query         : validator (mi_query_validator)
  - load_mi_semantics         : load the curated MI semantic registry
  - parse_user_question       : NL -> MIQuerySpec (deterministic + optional LLM)
  - execute_mi_query          : run a validated spec against canonical data
  - MIQueryResult             : executor result object
  - create_mi_chart           : render an MIQueryResult as a Plotly figure
  - MIChartResult             : chart factory result object
  - run_mi_agent_query        : question -> spec -> validate -> execute -> chart
  - parse_with_repair         : NL -> validated spec, with LLM repair loop
  - get_llm_config            : env-driven LLM configuration (cost control)

The curated semantic registry (mi_semantics_field_registry.yaml) is GENERATED
from the canonical field registry by build_mi_semantics_registry.py and is a
*reference* layer over canonical fields, not a copy of them.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .mi_query_spec import MIQuerySpec  # noqa: F401
from .mi_query_validator import (  # noqa: F401
    ValidationResult,
    load_mi_semantics,
    validate_mi_query,
)
from .mi_query_executor import (  # noqa: F401
    MIQueryExecutionError,
    MIQueryResult,
    execute_mi_query,
)
from .mi_chart_factory import (  # noqa: F401
    MIChartError,
    MIChartResult,
    create_mi_chart,
)
from .llm_query_parser import parse_user_question, parse_with_repair  # noqa: F401
from .mi_agent_config import LLMConfig, get_llm_config  # noqa: F401
from .mi_agent_workflow import run_mi_agent_query  # noqa: F401

__all__ = [
    "MIQuerySpec",
    "ValidationResult",
    "load_mi_semantics",
    "validate_mi_query",
    "execute_mi_query",
    "MIQueryResult",
    "MIQueryExecutionError",
    "create_mi_chart",
    "MIChartResult",
    "MIChartError",
    "parse_user_question",
    "parse_with_repair",
    "get_llm_config",
    "LLMConfig",
    "run_mi_agent_query",
    "__version__",
]
