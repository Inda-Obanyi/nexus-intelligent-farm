from typing import Optional

from nexus.core.memory.decision import DecisionRecord
from nexus.core.memory.experience import ExperienceRecord
from nexus.evaluation.decision_intelligence import (
    DecisionIntelligenceReport,
)


class ExperienceBuilder:
    """
    Convert NEXUS decision evidence into a learning experience.

    The builder does not make decisions.

    It does not modify DecisionRecord or DecisionIntelligenceReport.

    It simply creates a normalized ExperienceRecord that can be
    accumulated by ExperienceStore and analyzed later.
    """

    def build(
        self,
        record: DecisionRecord,
        intelligence: DecisionIntelligenceReport,
    ) -> ExperienceRecord:
        """
        Build an ExperienceRecord from existing NEXUS evidence.

        DecisionRecord provides:
            - state/context
            - selected action
            - execution
            - raw outcome
            - metadata

        DecisionIntelligenceReport provides:
            - confidence
            - alternatives
            - expected effect
            - observed effect
            - causal interpretation
            - evidence
        """

        if record.step != intelligence.step:
            raise ValueError(
                "DecisionRecord and DecisionIntelligenceReport "
                "must refer to the same step."
            )

        if record.decision_name != intelligence.decision_name:
            raise ValueError(
                "DecisionRecord and DecisionIntelligenceReport "
                "must refer to the same decision."
            )

        if list(record.decision_action) != list(
            intelligence.decision_action
        ):
            raise ValueError(
                "DecisionRecord and DecisionIntelligenceReport "
                "must contain the same decision action."
            )

        return ExperienceRecord(
            step=record.step,
            day=record.day,
            hour=record.hour,
            observation_summary=dict(
                record.observation_summary
            ),
            decision_name=record.decision_name,
            decision_action=list(
                record.decision_action
            ),
            decision_score=record.decision_score,
            confidence=intelligence.confidence,
            score_margin=intelligence.score_margin,
            runner_up=intelligence.runner_up,
            runner_up_score=intelligence.runner_up_score,
            alternative_count=intelligence.alternative_count,
            expected_effect=intelligence.expected_effect,
            observed_effect=intelligence.observed_effect,
            agent_effect=intelligence.agent_effect,
            environment_effect=intelligence.environment_effect,
            effect_match=intelligence.effect_match,
            executed_action=(
                dict(record.executed_action)
                if record.executed_action is not None
                else None
            ),
            outcome=(
                dict(record.outcome)
                if record.outcome is not None
                else None
            ),
            evidence=dict(intelligence.evidence),
            metadata=dict(record.metadata),
        )


def build_experience(
    record: DecisionRecord,
    intelligence: DecisionIntelligenceReport,
) -> ExperienceRecord:
    """
    Convenience function for building one experience record.
    """
    return ExperienceBuilder().build(
        record=record,
        intelligence=intelligence,
    )
