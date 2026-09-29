from dataclasses import dataclass
from typing import Any, Dict, List

from nexus.core.memory.experience_retriever import RetrievedExperience


@dataclass(frozen=True)
class EvidenceInterpretation:
    """
    Structured interpretation of retrieved historical evidence.

    This is descriptive only.

    It does not:
        - modify a decision score
        - select an action
        - rank current decisions
        - claim historical optimality
    """

    retrieved_count: int
    matched_count: int
    partial_count: int
    not_matched_count: int
    unknown_count: int

    dominant_outcome: str
    evidence_pattern: str

    supporting_steps: List[int]
    caution_steps: List[int]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "retrieved_count": self.retrieved_count,
            "matched_count": self.matched_count,
            "partial_count": self.partial_count,
            "not_matched_count": self.not_matched_count,
            "unknown_count": self.unknown_count,
            "dominant_outcome": self.dominant_outcome,
            "evidence_pattern": self.evidence_pattern,
            "supporting_steps": list(self.supporting_steps),
            "caution_steps": list(self.caution_steps),
        }


class EvidenceInterpreter:
    """
    Interpret retrieved experiences into descriptive historical patterns.

    The interpreter operates only on already-retrieved experiences.
    Similarity determines historical relevance, while effect_match
    determines the observed historical outcome.

    No current decision is changed here.
    """

    @staticmethod
    def _pattern(
        matched: int,
        partial: int,
        not_matched: int,
        unknown: int,
    ) -> str:
        known = matched + partial + not_matched

        if known == 0:
            return "NO_KNOWN_EVIDENCE"

        observed_types = sum(
            count > 0
            for count in (
                matched,
                partial,
                not_matched,
            )
        )

        if observed_types == 1:
            if matched == known:
                return "CONSISTENT_MATCH"

            if not_matched == known:
                return "CONSISTENT_MISMATCH"

            return "CONSISTENT_PARTIAL"

        return "MIXED_EVIDENCE"

    @classmethod
    def interpret(
        cls,
        retrieved: List[RetrievedExperience],
    ) -> EvidenceInterpretation:
        if not retrieved:
            return EvidenceInterpretation(
                retrieved_count=0,
                matched_count=0,
                partial_count=0,
                not_matched_count=0,
                unknown_count=0,
                dominant_outcome="NONE",
                evidence_pattern="NO_EVIDENCE",
                supporting_steps=[],
                caution_steps=[],
            )

        matched = 0
        partial = 0
        not_matched = 0
        unknown = 0

        for item in retrieved:
            effect_match = item.experience.effect_match

            if effect_match == "MATCHED":
                matched += 1
            elif effect_match == "PARTIAL":
                partial += 1
            elif effect_match == "NOT_MATCHED":
                not_matched += 1
            else:
                unknown += 1

        outcome_counts = {
            "MATCHED": matched,
            "PARTIAL": partial,
            "NOT_MATCHED": not_matched,
            "UNKNOWN": unknown,
        }

        dominant_outcome = max(
            outcome_counts,
            key=outcome_counts.get,
        )

        pattern = cls._pattern(
            matched,
            partial,
            not_matched,
            unknown,
        )

        ranked = sorted(
            retrieved,
            key=lambda item: item.similarity,
            reverse=True,
        )

        supporting_steps = [
            item.experience.step
            for item in ranked
            if item.experience.effect_match == "MATCHED"
        ][:3]

        caution_steps = [
            item.experience.step
            for item in ranked
            if item.experience.effect_match
            in {"PARTIAL", "NOT_MATCHED"}
        ][:3]

        return EvidenceInterpretation(
            retrieved_count=len(retrieved),
            matched_count=matched,
            partial_count=partial,
            not_matched_count=not_matched,
            unknown_count=unknown,
            dominant_outcome=dominant_outcome,
            evidence_pattern=pattern,
            supporting_steps=supporting_steps,
            caution_steps=caution_steps,
        )
