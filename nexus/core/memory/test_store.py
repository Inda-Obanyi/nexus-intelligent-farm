from nexus.core.memory.decision import DecisionRecord
from nexus.core.memory.store import DecisionMemory


def make_record(step=1):
    return DecisionRecord(
        step=step,
        day=1,
        hour=0,
        observation_summary={"money": 3000.0},
        decision_name="WATER_CROP",
        decision_action=["WATER"],
        decision_score=12.0,
        decision_reason="Crop requires water.",
        executed_action={
            "farmer": ["WATER"],
            "hands": [],
            "market": [],
        },
    )


def test_memory_stores_decision_record():
    memory = DecisionMemory()
    record = make_record()

    memory.add(record)

    assert len(memory) == 1
    assert memory.latest() is record


def test_memory_records_outcome_on_latest_decision():
    memory = DecisionMemory()
    memory.add(make_record())

    outcome = {
        "money_after": 2990.0,
        "crop_watered": True,
        "success": True,
    }

    memory.record_outcome(outcome)

    latest = memory.latest()

    assert latest is not None
    assert latest.outcome == outcome


def test_memory_copies_outcome_data():
    memory = DecisionMemory()
    memory.add(make_record())

    outcome = {
        "success": True,
        "value": 10,
    }

    memory.record_outcome(outcome)
    outcome["value"] = 999

    latest = memory.latest()

    assert latest is not None
    assert latest.outcome["value"] == 10


def test_memory_requires_existing_record_for_outcome():
    memory = DecisionMemory()

    try:
        memory.record_outcome({"success": True})
        assert False, "Expected ValueError."
    except ValueError as error:
        assert str(error) == (
            "Cannot record an outcome without a decision record."
        )


def test_memory_clear_removes_outcomes_and_decisions():
    memory = DecisionMemory()

    record = make_record()
    record.outcome = {"success": True}

    memory.add(record)
    assert len(memory) == 1

    memory.clear()

    assert len(memory) == 0
    assert memory.latest() is None


def test_memory_preserves_decision_trace():
    from nexus.core.decision.trace import CandidateScore, DecisionTrace

    trace = DecisionTrace.from_scored_candidates(
        [
            CandidateScore(
                name="WATER_CROP",
                score=12.0,
                action=["WATER"],
            ),
            CandidateScore(
                name="HARVEST_CROP",
                score=9.0,
                action=["HARVEST"],
            ),
        ]
    )

    record = make_record()
    record.decision_trace = trace

    memory = DecisionMemory()
    memory.add(record)

    latest = memory.latest()

    assert latest is not None
    assert latest.decision_trace is trace
    assert latest.decision_trace.selected_decision == "WATER_CROP"


def test_memory_preserves_decision_trace_metadata():
    from nexus.core.decision.trace import CandidateScore, DecisionTrace

    trace = DecisionTrace.from_scored_candidates(
        [
            CandidateScore(
                name="DIG_WEED",
                score=16.0,
                action=["DIG"],
            ),
            CandidateScore(
                name="WATER_CROP",
                score=9.0,
                action=["WATER"],
            ),
        ]
    )

    record = make_record()
    record.decision_trace = trace

    memory = DecisionMemory()
    memory.add(record)

    latest = memory.latest()

    assert latest is not None
    assert latest.decision_trace is not None
    assert latest.decision_trace.runner_up == "WATER_CROP"
    assert latest.decision_trace.score_margin == 7.0
    assert latest.decision_trace.candidate_count == 2
    assert latest.decision_trace.confidence == "HIGH"
