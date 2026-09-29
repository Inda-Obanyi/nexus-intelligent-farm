from dataclasses import dataclass
from typing import Any, Dict, List

from nexus.core.memory.experience_retriever import RetrievedExperience


@dataclass(frozen=True)
class LearningSignal:
    """
    Descriptive evidence aggregated from retrieved past experiences.

    This signal does not select a decision, modify a decision score,
    or claim that a historical decision was optimal.

    It answers only:

        "What does the retrieved historical evidence look like?"
    """

    retrieved_count: int
    matched_count: int
    partial_count: int
    not_matched_count: int
    unknown_count: int

    average_similarity: float
    top_similarity: float
    match_rate: float
    evidence_strength: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "retrieved_count": self.retrieved_count,
            "matched_count": self.matched_count,
            "partial_count": self.partial_count,
            "not_matched_count": self.not_matched_count,
            "unknown_count": self.unknown_count,
            "average_similarity": self.average_similarity,
            "top_similarity": self.top_similarity,
            "match_rate": self.match_rate,
            "evidence_strength": self.evidence_strength,
        }


class LearningSignalBuilder:
    """
    Build a bounded descriptive learning signal.

    No decision score is modified here.
    No recommendation is generated here.
    """

    @staticmethod
    def _evidence_strength(
        retrieved_count: int,
        average_similarity: float,
    ) -> str:
        if retrieved_count == 0:
            return "NONE"

        if retrieved_count >= 3 and average_similarity >= 0.75:
            return "STRONG"

        if retrieved_count >= 2 and average_similarity >= 0.50:
            return "MODERATE"

        return "WEAK"

    @classmethod
    def build(
        cls,
        retrieved: List[RetrievedExperience],
    ) -> LearningSignal:
        if not retrieved:
            return LearningSignal(
                retrieved_count=0,
                matched_count=0,
                partial_count=0,
                not_matched_count=0,
                unknown_count=0,
                average_similarity=0.0,
                top_similarity=0.0,
                match_rate=0.0,
                evidence_strength="NONE",
            )

        matched = 0
        partial = 0
        not_matched = 0
        unknown = 0

        similarities = []

        for item in retrieved:
            similarities.append(float(item.similarity))

            effect_match = item.experience.effect_match

            if effect_match == "MATCHED":
                matched += 1
            elif effect_match == "PARTIAL":
                partial += 1
            elif effect_match == "NOT_MATCHED":
                not_matched += 1
            else:
                unknown += 1

        average_similarity = (
            sum(similarities) / len(similarities)
        )

        known_count = (
            matched
            + partial
            + not_matched
        )

        match_rate = (
            matched / known_count
            if known_count > 0
            else 0.0
        )

        return LearningSignal(
            retrieved_count=len(retrieved),
            matched_count=matched,
            partial_count=partial,
            not_matched_count=not_matched,
            unknown_count=unknown,
            average_similarity=round(
                average_similarity,
                6,
            ),
            top_similarity=round(
                max(similarities),
                6,
            ),
            match_rate=round(
                match_rate,
                6,
            ),
            evidence_strength=cls._evidence_strength(
                len(retrieved),
                average_similarity,
            ),
        )
