from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from nexus.core.memory.experience import ExperienceRecord
from nexus.core.memory.experience_store import ExperienceStore


@dataclass(frozen=True)
class RetrievedExperience:
    """
    One contextually relevant past experience.

    The similarity score describes contextual similarity only.
    It is not a probability and does not imply that the past
    decision was optimal.
    """

    experience: ExperienceRecord
    similarity: float
    reasons: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "step": self.experience.step,
            "decision_name": self.experience.decision_name,
            "similarity": self.similarity,
            "reasons": list(self.reasons),
        }


class ExperienceRetriever:
    """
    Retrieve past experiences that resemble the current farm context.

    Retrieval is deliberately read-only and descriptive.

    IMPORTANT:
        Only decision-time context is used for similarity.

        Outcome fields such as:
            - effect_match
            - observed_effect
            - agent_effect
            - environment_effect
            - outcome
            - post-action evidence

        are never used to calculate similarity.
    """

    def __init__(
        self,
        store: ExperienceStore,
        max_results: int = 5,
    ) -> None:
        if not isinstance(store, ExperienceStore):
            raise TypeError(
                "store must be an ExperienceStore."
            )

        if max_results < 1:
            raise ValueError(
                "max_results must be at least 1."
            )

        self.store = store
        self.max_results = max_results

    @staticmethod
    def _distance(
        first: List[int],
        second: List[int],
    ) -> int:
        return (
            abs(int(first[0]) - int(second[0]))
            + abs(int(first[1]) - int(second[1]))
        )

    @staticmethod
    def _target_from_experience(
        experience: ExperienceRecord,
    ) -> Optional[List[int]]:
        """
        Extract a spatial target recorded at decision time.

        Target metadata may exist either directly in metadata or
        inside execution_intent metadata.
        """

        metadata = experience.metadata

        target = metadata.get("planning_target")

        if isinstance(target, dict):
            if "x" in target and "y" in target:
                return [
                    int(target["x"]),
                    int(target["y"]),
                ]

        intent = metadata.get("execution_intent")

        if isinstance(intent, dict):
            target = intent.get("target")

            if isinstance(target, dict):
                if "x" in target and "y" in target:
                    return [
                        int(target["x"]),
                        int(target["y"]),
                    ]

        return None

    @classmethod
    def _context_features(
        cls,
        experience: ExperienceRecord,
    ) -> Dict[str, Any]:
        """
        Extract only contextual features that existed around
        the decision event.

        No outcome/effect fields are included.
        """

        observation = experience.observation_summary

        position = observation.get(
            "farmer_position"
        )

        return {
            "decision_name": experience.decision_name,
            "decision_action": tuple(
                experience.decision_action
            ),
            "money": float(
                observation.get("money", 0.0)
            ),
            "crop_count": int(
                observation.get("crop_count", 0)
            ),
            "weed_count": int(
                observation.get("weed_count", 0)
            ),
            "wheat_seeds": int(
                observation.get("wheat_seeds", 0)
            ),
            "wheat_inventory": int(
                observation.get("wheat_inventory", 0)
            ),
            "farmer_position": (
                list(position)
                if isinstance(position, (list, tuple))
                and len(position) >= 2
                else None
            ),
            "spatial_target": cls._target_from_experience(
                experience
            ),
        }

    @classmethod
    def _similarity(
        cls,
        current: Dict[str, Any],
        past: ExperienceRecord,
    ) -> tuple[float, List[str]]:
        """
        Calculate a bounded contextual similarity score.

        Maximum score = 1.0.

        Decision identity is the strongest signal.
        Numeric/contextual similarities provide additional evidence.
        """

        previous = cls._context_features(past)

        score = 0.0
        reasons: List[str] = []

        # -----------------------------------------------------
        # Decision identity
        # -----------------------------------------------------

        if (
            current["decision_name"]
            == previous["decision_name"]
        ):
            score += 0.40
            reasons.append(
                "same decision type"
            )

        # -----------------------------------------------------
        # Action identity
        # -----------------------------------------------------

        if (
            current["decision_action"]
            == previous["decision_action"]
        ):
            score += 0.10
            reasons.append(
                "same decision action"
            )

        # -----------------------------------------------------
        # Farm population context
        # -----------------------------------------------------

        if (
            current["crop_count"]
            == previous["crop_count"]
        ):
            score += 0.10
            reasons.append(
                "same crop count"
            )

        if (
            current["weed_count"]
            == previous["weed_count"]
        ):
            score += 0.10
            reasons.append(
                "same weed count"
            )

        # -----------------------------------------------------
        # Resource context
        # -----------------------------------------------------

        if (
            current["wheat_seeds"]
            == previous["wheat_seeds"]
        ):
            score += 0.05
            reasons.append(
                "same wheat seed level"
            )

        if (
            current["wheat_inventory"]
            == previous["wheat_inventory"]
        ):
            score += 0.05
            reasons.append(
                "same wheat inventory level"
            )

        # -----------------------------------------------------
        # Economic context
        # -----------------------------------------------------

        current_money = current["money"]
        previous_money = previous["money"]

        money_scale = max(
            abs(current_money),
            abs(previous_money),
            1.0,
        )

        money_difference = abs(
            current_money - previous_money
        )

        if money_difference <= money_scale * 0.10:
            score += 0.05
            reasons.append(
                "similar cash position"
            )

        # -----------------------------------------------------
        # Spatial context
        # -----------------------------------------------------

        current_position = current[
            "farmer_position"
        ]
        previous_position = previous[
            "farmer_position"
        ]

        if (
            current_position is not None
            and previous_position is not None
        ):
            distance = cls._distance(
                current_position,
                previous_position,
            )

            if distance == 0:
                score += 0.05
                reasons.append(
                    "same farmer position"
                )
            elif distance <= 2:
                score += 0.025
                reasons.append(
                    "nearby farmer position"
                )

        current_target = current[
            "spatial_target"
        ]
        previous_target = previous[
            "spatial_target"
        ]

        if (
            current_target is not None
            and previous_target is not None
        ):
            distance = cls._distance(
                current_target,
                previous_target,
            )

            if distance == 0:
                score += 0.05
                reasons.append(
                    "same spatial target"
                )
            elif distance <= 2:
                score += 0.025
                reasons.append(
                    "nearby spatial target"
                )

        return round(
            min(score, 1.0),
            6,
        ), reasons

    def retrieve(
        self,
        observation_summary: Dict[str, Any],
        decision_name: str,
        decision_action: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedExperience]:
        """
        Retrieve the most contextually relevant past experiences.

        This method does not modify the store and does not alter
        decision selection.
        """

        metadata = metadata or {}

        current = {
            "decision_name": decision_name,
            "decision_action": tuple(
                decision_action
            ),
            "money": float(
                observation_summary.get(
                    "money",
                    0.0,
                )
            ),
            "crop_count": int(
                observation_summary.get(
                    "crop_count",
                    0,
                )
            ),
            "weed_count": int(
                observation_summary.get(
                    "weed_count",
                    0,
                )
            ),
            "wheat_seeds": int(
                observation_summary.get(
                    "wheat_seeds",
                    0,
                )
            ),
            "wheat_inventory": int(
                observation_summary.get(
                    "wheat_inventory",
                    0,
                )
            ),
            "farmer_position": (
                list(
                    observation_summary[
                        "farmer_position"
                    ]
                )
                if isinstance(
                    observation_summary.get(
                        "farmer_position"
                    ),
                    (list, tuple),
                )
                and len(
                    observation_summary[
                        "farmer_position"
                    ]
                ) >= 2
                else None
            ),
            "spatial_target": None,
        }

        planning_target = metadata.get(
            "planning_target"
        )

        if isinstance(planning_target, dict):
            if (
                "x" in planning_target
                and "y" in planning_target
            ):
                current["spatial_target"] = [
                    int(planning_target["x"]),
                    int(planning_target["y"]),
                ]

        retrieved: List[RetrievedExperience] = []

        for experience in self.store.all():
            # -------------------------------------------------
            # Decision identity is a retrieval boundary.
            #
            # Experiences from a different strategic decision
            # are not treated as evidence for the current one,
            # even when their farm context happens to be similar.
            # -------------------------------------------------

            if (
                experience.decision_name
                != decision_name
            ):
                continue

            similarity, reasons = (
                self._similarity(
                    current,
                    experience,
                )
            )

            if similarity <= 0:
                continue

            retrieved.append(
                RetrievedExperience(
                    experience=experience,
                    similarity=similarity,
                    reasons=reasons,
                )
            )

        retrieved.sort(
            key=lambda item: (
                -item.similarity,
                -item.experience.step,
            )
        )

        return retrieved[: self.max_results]

    def summarize(
        self,
        retrieved: List[RetrievedExperience],
    ) -> Dict[str, Any]:
        """
        Produce bounded descriptive evidence from retrieval.

        This does not recommend a decision and does not modify
        scores.
        """

        if not retrieved:
            return {
                "retrieved_count": 0,
                "top_similarity": 0.0,
                "matched_decisions": [],
                "evidence": [],
            }

        return {
            "retrieved_count": len(retrieved),
            "top_similarity": retrieved[0].similarity,
            "matched_decisions": [
                item.experience.decision_name
                for item in retrieved
            ],
            "evidence": [
                {
                    "step": item.experience.step,
                    "decision_name": (
                        item.experience.decision_name
                    ),
                    "similarity": item.similarity,
                    "effect_match": (
                        item.experience.effect_match
                    ),
                    "reasons": list(item.reasons),
                }
                for item in retrieved
            ],
        }
