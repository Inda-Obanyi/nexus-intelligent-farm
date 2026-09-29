from nexus.core.planning.execution_intent import ExecutionIntent


def test_execution_intent_starts_in_progress():
    intent = ExecutionIntent(
        decision_name="PLANT_CARROT",
        action=["PLANT", "CARROT"],
        target={"x": 4, "y": 3},
    )

    assert intent.status == "IN_PROGRESS"
    assert intent.is_active()
    assert not intent.is_completed()
    assert not intent.is_failed()


def test_execution_intent_preserves_target():
    intent = ExecutionIntent(
        decision_name="CLEAR_WEED",
        action=["CLEAR_WEED", 4, 3],
        target={"x": 4, "y": 3},
    )

    assert intent.target == {
        "x": 4,
        "y": 3,
    }


def test_execution_intent_records_steps():
    intent = ExecutionIntent(
        decision_name="PLANT_CARROT",
        action=["PLANT", "CARROT"],
        target={"x": 4, "y": 3},
    )

    intent.record_step(["NORTH"])
    intent.record_step(["NORTH"])
    intent.record_step(["PLANT", "CARROT"])

    assert intent.steps == [
        ["NORTH"],
        ["NORTH"],
        ["PLANT", "CARROT"],
    ]

    assert intent.execution_steps == 3


def test_execution_intent_can_complete():
    intent = ExecutionIntent(
        decision_name="PLANT_CARROT",
        action=["PLANT", "CARROT"],
        target={"x": 4, "y": 3},
    )

    intent.complete()

    assert intent.status == "COMPLETED"
    assert intent.is_completed()
    assert not intent.is_active()


def test_execution_intent_can_fail():
    intent = ExecutionIntent(
        decision_name="CLEAR_WEED",
        action=["CLEAR_WEED", 4, 3],
        target={"x": 4, "y": 3},
    )

    intent.fail()

    assert intent.status == "FAILED"
    assert intent.is_failed()
    assert not intent.is_active()


def test_execution_intent_as_dict():
    intent = ExecutionIntent(
        decision_name="CLEAR_WEED",
        action=["CLEAR_WEED", 4, 3],
        target={"x": 4, "y": 3},
    )

    intent.record_step(["WEST"])

    data = intent.as_dict()

    assert data["decision_name"] == "CLEAR_WEED"
    assert data["action"] == ["CLEAR_WEED", 4, 3]
    assert data["target"] == {"x": 4, "y": 3}
    assert data["status"] == "IN_PROGRESS"
    assert data["steps"] == [["WEST"]]
    assert data["execution_steps"] == 1