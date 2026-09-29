from typing import Any, Dict, List

from nexus.core.memory.decision import DecisionRecord


class DecisionMemory:
    """
    In-memory store for NEXUS decision history.

    Records the sequence of decisions made by an agent
    during an episode or session.
    """

    def __init__(self):
        self.records: List[DecisionRecord] = []

    def add(self, record: DecisionRecord) -> None:
        """Store a decision record."""
        self.records.append(record)

    def all(self) -> List[DecisionRecord]:
        """Return all stored decision records."""
        return list(self.records)

    def latest(self) -> DecisionRecord | None:
        """Return the most recent decision, if available."""
        if not self.records:
            return None
        return self.records[-1]

    def record_outcome(
        self,
        outcome: Dict[str, Any],
    ) -> None:
        """
        Attach an observed outcome to the most recent decision.

        The outcome represents what happened after the decision
        was executed in the environment.
        """
        latest_record = self.latest()

        if latest_record is None:
            raise ValueError(
                "Cannot record an outcome without a decision record."
            )

        latest_record.outcome = dict(outcome)

    def clear(self) -> None:
        """Clear the decision history."""
        self.records.clear()

    def __len__(self) -> int:
        return len(self.records)
