from nexus.core.memory.evidence_interpreter import EvidenceInterpreter
from nexus.core.memory.experience import ExperienceRecord
from nexus.core.memory.experience_retriever import RetrievedExperience


def make_retrieved(step, similarity, effect_match):
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
        decision_action=["BUY_SEED", "WHEAT", 1],
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


def test_empty_retrieval_has_no_evidence():
    result = EvidenceInterpreter.interpret([])

    assert result.retrieved_count == 0
    assert result.dominant_outcome == "NONE"
    assert result.evidence_pattern == "NO_EVIDENCE"
    assert result.supporting_steps == []
    assert result.caution_steps == []


def test_consistent_matches_are_match_dominant():
    retrieved = [
        make_retrieved(1, 0.95, "MATCHED"),
        make_retrieved(2, 0.85, "MATCHED"),
        make_retrieved(3, 0.75, "MATCHED"),
    ]

    result = EvidenceInterpreter.interpret(retrieved)

    assert result.dominant_outcome == "MATCHED"
    assert result.evidence_pattern == "CONSISTENT_MATCH"
    assert result.supporting_steps == [1, 2, 3]
    assert result.caution_steps == []


def test_mixed_evidence_identifies_support_and_caution_steps():
    retrieved = [
        make_retrieved(10, 0.95, "MATCHED"),
        make_retrieved(20, 0.90, "PARTIAL"),
        make_retrieved(30, 0.80, "NOT_MATCHED"),
        make_retrieved(40, 0.70, "MATCHED"),
    ]

    result = EvidenceInterpreter.interpret(retrieved)

    assert result.evidence_pattern == "MIXED_EVIDENCE"
    assert result.dominant_outcome == "MATCHED"
    assert result.supporting_steps == [10, 40]
    assert result.caution_steps == [20, 30]


def test_unknown_evidence_is_preserved():
    retrieved = [
        make_retrieved(1, 0.90, ""),
        make_retrieved(2, 0.80, ""),
    ]

    result = EvidenceInterpreter.interpret(retrieved)

    assert result.unknown_count == 2
    assert result.dominant_outcome == "UNKNOWN"
    assert result.evidence_pattern == "NO_KNOWN_EVIDENCE"


def test_as_dict_is_serializable_and_descriptive():
    retrieved = [
        make_retrieved(5, 0.90, "MATCHED"),
        make_retrieved(6, 0.80, "PARTIAL"),
    ]

    result = EvidenceInterpreter.interpret(retrieved)
    payload = result.as_dict()

    assert payload["retrieved_count"] == 2
    assert payload["matched_count"] == 1
    assert payload["partial_count"] == 1
    assert payload["supporting_steps"] == [5]
    assert payload["caution_steps"] == [6]
