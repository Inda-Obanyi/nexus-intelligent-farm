from nexus.core.decision.trace import (
    CandidateScore,
    DecisionTrace,
)
from nexus.core.memory.decision import DecisionRecord
from nexus.core.memory.experience import ExperienceRecord
from nexus.core.memory.experience_builder import ExperienceBuilder
from nexus.core.memory.experience_store import ExperienceStore
from nexus.evaluation.decision_intelligence import (
    DecisionIntelligenceBuilder,
)
from nexus.evaluation.outcomes import OutcomeEvaluation


def make_record() -> DecisionRecord:
    trace = DecisionTrace.from_scored_candidates(
        [
            CandidateScore(
                name="WATER_CROP",
                score=8.0,
                action=["WATER_CROP"],
            ),
            CandidateScore(
                name="PASS",
                score=5.0,
                action=["PASS"],
            ),
        ]
    )

    return DecisionRecord(
        step=10,
        day=1,
        hour=10,
        observation_summary={
            "money": 2800.0,
            "farmer_position": [4, 4],
            "crop_count": 1,
            "weed_count": 0,
        },
        decision_name="WATER_CROP",
        decision_action=["WATER_CROP"],
        decision_score=8.0,
        decision_reason="Water the active crop.",
        decision_trace=trace,
        executed_action={
            "farmer": ["WATER"],
            "hands": [],
            "market": [],
        },
        metadata={
            "planning_target": None,
        },
    )


def make_outcome() -> OutcomeEvaluation:
    return OutcomeEvaluation(
        decision_name="WATER_CROP",
        executed=True,
        execution_effect="CROP_WATERED",
        money_delta=0.0,
        economic_effect="NO_MONEY_CHANGE",
        position_changed=False,
        crop_count_delta=0,
        weed_count_delta=0,
        operational_effect="CROP_WATERED",
        expected_effect="Water the active crop.",
        observed_effect="Crop was watered.",
        agent_effect="Water action was executed.",
        environment_effect="No exogenous effect observed.",
        effect_match="MATCHED",
        evidence={
            "watered_before": False,
            "watered_after": True,
        },
    )


def make_intelligence() -> object:
    record = make_record()
    outcome = make_outcome()

    return DecisionIntelligenceBuilder().build(
        record=record,
        outcome=outcome,
    )


def test_experience_builder_creates_experience():
    record = make_record()
    intelligence = make_intelligence()

    experience = ExperienceBuilder().build(
        record=record,
        intelligence=intelligence,
    )

    assert isinstance(experience, ExperienceRecord)
    assert experience.step == 10
    assert experience.decision_name == "WATER_CROP"
    assert experience.decision_action == ["WATER_CROP"]
    assert experience.confidence == "MEDIUM"
    assert experience.score_margin == 3.0
    assert experience.runner_up == "PASS"
    assert experience.alternative_count == 1
    assert experience.effect_match == "MATCHED"
    assert experience.has_observed_outcome is True
    assert experience.effect_was_matched is True


def test_experience_record_serializes():
    record = make_record()
    intelligence = make_intelligence()

    experience = ExperienceBuilder().build(
        record=record,
        intelligence=intelligence,
    )

    data = experience.as_dict()

    assert data["step"] == 10
    assert data["decision_name"] == "WATER_CROP"
    assert data["decision_action"] == ["WATER_CROP"]
    assert data["confidence"] == "MEDIUM"
    assert data["runner_up"] == "PASS"
    assert data["alternative_count"] == 1
    assert data["effect_match"] == "MATCHED"


def test_experience_builder_rejects_mismatched_step():
    record = make_record()
    intelligence = make_intelligence()

    mismatched = type(intelligence)(
        step=11,
        day=intelligence.day,
        hour=intelligence.hour,
        decision_name=intelligence.decision_name,
        decision_action=list(
            intelligence.decision_action
        ),
        decision_score=intelligence.decision_score,
        decision_reason=intelligence.decision_reason,
        confidence=intelligence.confidence,
        score_margin=intelligence.score_margin,
        runner_up=intelligence.runner_up,
        runner_up_score=intelligence.runner_up_score,
        alternatives=list(intelligence.alternatives),
        expected_effect=intelligence.expected_effect,
        observed_effect=intelligence.observed_effect,
        agent_effect=intelligence.agent_effect,
        environment_effect=intelligence.environment_effect,
        effect_match=intelligence.effect_match,
        evidence=dict(intelligence.evidence),
        executed_action=intelligence.executed_action,
    )

    try:
        ExperienceBuilder().build(
            record=record,
            intelligence=mismatched,
        )
    except ValueError as exc:
        assert "same step" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for mismatched step."
        )


def test_experience_store_add_and_retrieve():
    record = make_record()
    intelligence = make_intelligence()

    experience = ExperienceBuilder().build(
        record=record,
        intelligence=intelligence,
    )

    store = ExperienceStore()
    store.add(experience)

    assert len(store) == 1
    assert store.latest() == experience
    assert store.all() == [experience]


def test_experience_store_filters_by_decision():
    record = make_record()
    intelligence = make_intelligence()

    experience = ExperienceBuilder().build(
        record=record,
        intelligence=intelligence,
    )

    store = ExperienceStore()
    store.add(experience)

    assert store.by_decision("WATER_CROP") == [
        experience
    ]
    assert store.by_decision("PASS") == []


def test_experience_store_classifies_effects():
    record = make_record()
    intelligence = make_intelligence()

    experience = ExperienceBuilder().build(
        record=record,
        intelligence=intelligence,
    )

    store = ExperienceStore()
    store.add(experience)

    assert store.matched() == [experience]
    assert store.partial() == []
    assert store.mismatched() == []
    assert store.unknown() == []


def test_experience_store_counts_decisions():
    record = make_record()
    intelligence = make_intelligence()

    experience = ExperienceBuilder().build(
        record=record,
        intelligence=intelligence,
    )

    store = ExperienceStore()
    store.add_many(
        [
            experience,
            experience,
        ]
    )

    assert store.decision_counts() == {
        "WATER_CROP": 2
    }


def test_experience_store_clear():
    record = make_record()
    intelligence = make_intelligence()

    experience = ExperienceBuilder().build(
        record=record,
        intelligence=intelligence,
    )

    store = ExperienceStore()
    store.add(experience)

    assert len(store) == 1

    store.clear()

    assert len(store) == 0
    assert store.latest() is None
