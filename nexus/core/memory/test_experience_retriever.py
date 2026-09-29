from nexus.core.memory.decision import DecisionRecord
from nexus.core.memory.experience import ExperienceRecord
from nexus.core.memory.experience_retriever import (
    ExperienceRetriever,
    RetrievedExperience,
)
from nexus.core.memory.experience_store import ExperienceStore


def make_experience(
    step=1,
    decision_name="WATER_CROP",
    decision_action=None,
    position=None,
    effect_match="MATCHED",
    metadata=None,
):
    if decision_action is None:
        decision_action = ["WATER_CROP"]

    if position is None:
        position = [4, 4]

    return ExperienceRecord(
        step=step,
        day=1,
        hour=10,
        observation_summary={
            "money": 2800.0,
            "farmer_position": position,
            "crop_count": 1,
            "weed_count": 0,
            "wheat_seeds": 2,
            "wheat_inventory": 0,
        },
        decision_name=decision_name,
        decision_action=decision_action,
        decision_score=8.0,
        confidence="MEDIUM",
        score_margin=3.0,
        expected_effect="Water the active crop.",
        observed_effect="Crop was watered.",
        effect_match=effect_match,
        outcome={
            "money_after": 2800.0,
            "secret_test_value": "must_not_affect_similarity",
        },
        evidence={
            "secret_evidence_value": 999999,
        },
        metadata=metadata or {},
    )


def test_retriever_returns_contextually_similar_experience():
    store = ExperienceStore()

    experience = make_experience(
        step=10,
        metadata={
            "planning_target": {
                "x": 4,
                "y": 4,
            }
        },
    )

    store.add(experience)

    retriever = ExperienceRetriever(
        store=store,
        max_results=5,
    )

    results = retriever.retrieve(
        observation_summary={
            "money": 2810.0,
            "farmer_position": [4, 4],
            "crop_count": 1,
            "weed_count": 0,
            "wheat_seeds": 2,
            "wheat_inventory": 0,
        },
        decision_name="WATER_CROP",
        decision_action=["WATER_CROP"],
        metadata={
            "planning_target": {
                "x": 4,
                "y": 4,
            }
        },
    )

    assert len(results) == 1
    assert isinstance(
        results[0],
        RetrievedExperience,
    )
    assert results[0].experience is experience
    assert results[0].similarity > 0.9


def test_retriever_prioritizes_same_decision_type():
    store = ExperienceStore()

    water = make_experience(
        step=1,
        decision_name="WATER_CROP",
    )

    harvest = make_experience(
        step=2,
        decision_name="HARVEST_CROP",
        decision_action=["HARVEST_CROP"],
    )

    store.add_many([harvest, water])

    retriever = ExperienceRetriever(store)

    results = retriever.retrieve(
        observation_summary=water.observation_summary,
        decision_name="WATER_CROP",
        decision_action=["WATER_CROP"],
    )

    assert results[0].experience is water


def test_retriever_uses_approximate_spatial_similarity():
    store = ExperienceStore()

    experience = make_experience(
        position=[5, 4],
        metadata={
            "planning_target": {
                "x": 5,
                "y": 4,
            }
        },
    )

    store.add(experience)

    retriever = ExperienceRetriever(store)

    results = retriever.retrieve(
        observation_summary={
            "money": 2800.0,
            "farmer_position": [4, 4],
            "crop_count": 1,
            "weed_count": 0,
            "wheat_seeds": 2,
            "wheat_inventory": 0,
        },
        decision_name="WATER_CROP",
        decision_action=["WATER_CROP"],
        metadata={
            "planning_target": {
                "x": 4,
                "y": 4,
            }
        },
    )

    assert len(results) == 1
    assert results[0].similarity > 0


def test_retriever_does_not_use_outcome_for_similarity():
    store = ExperienceStore()

    matched = make_experience(
        step=1,
        effect_match="MATCHED",
    )

    mismatched = make_experience(
        step=2,
        effect_match="MISMATCH",
    )

    store.add_many(
        [
            matched,
            mismatched,
        ]
    )

    retriever = ExperienceRetriever(store)

    results = retriever.retrieve(
        observation_summary=matched.observation_summary,
        decision_name="WATER_CROP",
        decision_action=["WATER_CROP"],
    )

    assert len(results) == 2
    assert (
        results[0].similarity
        == results[1].similarity
    )


def test_retriever_returns_empty_for_unrelated_decision():
    store = ExperienceStore()

    store.add(
        make_experience(
            decision_name="WATER_CROP",
        )
    )

    retriever = ExperienceRetriever(store)

    results = retriever.retrieve(
        observation_summary={
            "money": 2800.0,
            "farmer_position": [4, 4],
            "crop_count": 1,
            "weed_count": 0,
            "wheat_seeds": 2,
            "wheat_inventory": 0,
        },
        decision_name="SELL_WHEAT",
        decision_action=["SELL", "WHEAT"],
    )

    assert results == []


def test_retriever_limits_results():
    store = ExperienceStore()

    for step in range(1, 8):
        store.add(
            make_experience(step=step)
        )

    retriever = ExperienceRetriever(
        store=store,
        max_results=3,
    )

    results = retriever.retrieve(
        observation_summary={
            "money": 2800.0,
            "farmer_position": [4, 4],
            "crop_count": 1,
            "weed_count": 0,
            "wheat_seeds": 2,
            "wheat_inventory": 0,
        },
        decision_name="WATER_CROP",
        decision_action=["WATER_CROP"],
    )

    assert len(results) == 3


def test_retriever_summary_is_descriptive():
    experience = make_experience()

    retrieved = [
        RetrievedExperience(
            experience=experience,
            similarity=0.95,
            reasons=[
                "same decision type",
                "same farmer position",
            ],
        )
    ]

    summary = ExperienceRetriever(
        ExperienceStore()
    ).summarize(retrieved)

    assert summary["retrieved_count"] == 1
    assert summary["top_similarity"] == 0.95
    assert summary["matched_decisions"] == [
        "WATER_CROP"
    ]
    assert summary["evidence"][0][
        "effect_match"
    ] == "MATCHED"
