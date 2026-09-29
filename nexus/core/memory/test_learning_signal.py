from nexus.core.memory.experience import ExperienceRecord
from nexus.core.memory.experience_retriever import (
    RetrievedExperience,
)
from nexus.core.memory.learning_signal import (
    LearningSignalBuilder,
)


def make_retrieved(
    step,
    similarity,
    effect_match,
):
    experience = ExperienceRecord(
        step=step,
        day=1,
        hour=1,
        observation_summary={
            "money": 100.0,
            "crop_count": 2,
            "weed_count": 1,
            "wheat_seeds": 3,
            "wheat_inventory": 0,
            "farmer_position": [2, 2],
        },
        decision_name="BUY_WHEAT_SEED",
        decision_action=[
            "BUY_SEED",
            "WHEAT",
            1,
        ],
        decision_score=10.0,
        confidence="HIGH",
        score_margin=2.0,
        effect_match=effect_match,
    )

    return RetrievedExperience(
        experience=experience,
        similarity=similarity,
        reasons=["same decision type"],
    )


def test_empty_retrieval_produces_no_signal():
    signal = LearningSignalBuilder.build([])

    assert signal.retrieved_count == 0
    assert signal.matched_count == 0
    assert signal.match_rate == 0.0
    assert signal.evidence_strength == "NONE"


def test_signal_counts_outcomes():
    retrieved = [
        make_retrieved(1, 0.9, "MATCHED"),
        make_retrieved(2, 0.8, "MATCHED"),
        make_retrieved(3, 0.7, "PARTIAL"),
        make_retrieved(4, 0.6, "NOT_MATCHED"),
        make_retrieved(5, 0.5, ""),
    ]

    signal = LearningSignalBuilder.build(retrieved)

    assert signal.retrieved_count == 5
    assert signal.matched_count == 2
    assert signal.partial_count == 1
    assert signal.not_matched_count == 1
    assert signal.unknown_count == 1


def test_match_rate_uses_known_outcomes_only():
    retrieved = [
        make_retrieved(1, 0.9, "MATCHED"),
        make_retrieved(2, 0.8, "MATCHED"),
        make_retrieved(3, 0.7, ""),
    ]

    signal = LearningSignalBuilder.build(retrieved)

    assert signal.match_rate == 1.0


def test_similarity_statistics_are_bounded():
    retrieved = [
        make_retrieved(1, 0.9, "MATCHED"),
        make_retrieved(2, 0.7, "PARTIAL"),
        make_retrieved(3, 0.5, "NOT_MATCHED"),
    ]

    signal = LearningSignalBuilder.build(retrieved)

    assert signal.top_similarity == 0.9
    assert signal.average_similarity == 0.7


def test_strong_evidence_requires_multiple_similar_experiences():
    retrieved = [
        make_retrieved(1, 0.95, "MATCHED"),
        make_retrieved(2, 0.85, "MATCHED"),
        make_retrieved(3, 0.80, "PARTIAL"),
    ]

    signal = LearningSignalBuilder.build(retrieved)

    assert signal.evidence_strength == "STRONG"


def test_signal_is_descriptive_only():
    retrieved = [
        make_retrieved(1, 0.9, "NOT_MATCHED"),
        make_retrieved(2, 0.8, "NOT_MATCHED"),
    ]

    signal = LearningSignalBuilder.build(retrieved)

    assert signal.retrieved_count == 2
    assert signal.not_matched_count == 2
    assert signal.evidence_strength == "MODERATE"
