from dataclasses import dataclass
from typing import List


@dataclass
class DecisionOption:
    """A possible action considered by the NEXUS decision engine."""

    name: str
    action: List[str]
    score: float
    reason: str
