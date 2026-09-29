from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nexus.core.decision.trace import DecisionTrace


@dataclass
class DecisionRecord:
    """
    Immutable-style record of a NEXUS decision cycle.

    Captures what NEXUS perceived, considered, decided, executed,
    and eventually what happened as a result.
    """

    step: int
    day: int
    hour: int

    observation_summary: Dict[str, Any]

    decision_name: str
    decision_action: List[str]
    decision_score: float
    decision_reason: str

    decision_trace: Optional[DecisionTrace] = None

    executed_action: Optional[Dict[str, Any]] = None

    outcome: Optional[Dict[str, Any]] = None

    metadata: Dict[str, Any] = field(default_factory=dict)
