from typing import Dict, Iterable, List, Optional

from nexus.core.memory.experience import ExperienceRecord


class ExperienceStore:
    """
    In-memory store for NEXUS learning experiences.

    DecisionMemory stores decision events.

    ExperienceStore stores normalized learning-oriented experiences
    derived from those decision events and their observed outcomes.

    Keeping the two stores separate allows the decision/audit history
    to remain stable while the learning infrastructure evolves.
    """

    def __init__(self):
        self.records: List[ExperienceRecord] = []

    def add(self, experience: ExperienceRecord) -> None:
        """Store one learning experience."""
        if not isinstance(experience, ExperienceRecord):
            raise TypeError(
                "experience must be an ExperienceRecord."
            )

        self.records.append(experience)

    def add_many(
        self,
        experiences: Iterable[ExperienceRecord],
    ) -> None:
        """Store multiple learning experiences."""
        for experience in experiences:
            self.add(experience)

    def all(self) -> List[ExperienceRecord]:
        """Return all stored experiences."""
        return list(self.records)

    def latest(self) -> Optional[ExperienceRecord]:
        """Return the most recent experience, if available."""
        if not self.records:
            return None

        return self.records[-1]

    def by_decision(
        self,
        decision_name: str,
    ) -> List[ExperienceRecord]:
        """
        Return all experiences associated with one decision type.
        """
        return [
            record
            for record in self.records
            if record.decision_name == decision_name
        ]

    def matched(self) -> List[ExperienceRecord]:
        """Return experiences whose expected effect was matched."""
        return [
            record
            for record in self.records
            if record.effect_match == "MATCHED"
        ]

    def partial(self) -> List[ExperienceRecord]:
        """Return experiences with a partial effect."""
        return [
            record
            for record in self.records
            if record.effect_match == "PARTIAL"
        ]

    def mismatched(self) -> List[ExperienceRecord]:
        """Return experiences whose expected effect did not match."""
        return [
            record
            for record in self.records
            if record.effect_match == "MISMATCH"
        ]

    def unknown(self) -> List[ExperienceRecord]:
        """Return experiences with unknown effect classification."""
        return [
            record
            for record in self.records
            if record.effect_match == "UNKNOWN"
        ]

    def decision_counts(self) -> Dict[str, int]:
        """
        Count how many experiences exist for each decision type.
        """
        counts: Dict[str, int] = {}

        for record in self.records:
            counts[record.decision_name] = (
                counts.get(record.decision_name, 0) + 1
            )

        return counts

    def clear(self) -> None:
        """Remove all stored experiences."""
        self.records.clear()

    def __len__(self) -> int:
        """Return the number of stored experiences."""
        return len(self.records)
