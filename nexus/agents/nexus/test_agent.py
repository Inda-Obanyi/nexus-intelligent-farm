from nexus.agents.nexus.agent import NexusAgent


class ContextCapturingEngine:
    def __init__(self):
        self.context = None

    def decide(self, context):
        self.context = context

        class Decision:
            name = "PASS"
            action = ["PASS"]
            score = 0.0
            reason = "Test decision."

        return Decision()


class FixedDecisionEngine:
    def __init__(
        self,
        name,
        action,
        score=10.0,
        reason="Test decision.",
    ):
        self.name = name
        self.action = action
        self.score = score
        self.reason = reason
        self.last_trace = None

    def decide(self, context):
        class Decision:
            pass

        decision = Decision()
        decision.name = self.name
        decision.action = list(self.action)
        decision.score = self.score
        decision.reason = self.reason

        return decision


def make_observation(
    farmer=(0, 0),
    tiles=None,
    seeds=None,
    money=3000,
):
    if tiles is None:
        tiles = [
            [None]
        ]

    if seeds is None:
        seeds = {}

    return {
        "step": 1,
        "day": 1,
        "hour": 0,
        "player": 0,
        "farms": [
            {
                "money": money,
                "farmer": list(farmer),
                "tiles": tiles,
            }
        ],
        "private": {
            "seeds": dict(seeds),
            "shed": {},
        },
        "market": {
            "prices": {},
            "inventory": {},
        },
    }


def test_agent_builds_crop_risk_before_decision():
    agent = NexusAgent()

    engine = ContextCapturingEngine()
    agent.decision_engine = engine

    observation = make_observation(
        farmer=(0, 0),
        tiles=[
            [
                {
                    "kind": "PLANT",
                    "crop": "WHEAT",
                    "planted_day": 0,
                    "watered_today": False,
                    "yield_units": 0,
                    "fertilized_until_day": -1,
                }
            ]
        ],
    )

    action = agent.act(observation)

    assert action["farmer"] == ["PASS"]
    assert action["hands"] == []
    assert action["market"] == []

    assert engine.context is not None
    assert "WHEAT" in engine.context.crop_risks

    wheat_risk = engine.context.crop_risks["WHEAT"]

    assert wheat_risk.crop == "WHEAT"
    assert wheat_risk.operational == 0.5


def test_agent_persists_decision_trace_in_memory():
    agent = NexusAgent()

    action = agent.act(
        make_observation(
            farmer=(0, 0),
            tiles=[
                [
                    {
                        "kind": "PLANT",
                        "crop": "WHEAT",
                        "planted_day": 0,
                        "watered_today": False,
                        "yield_units": 0,
                        "fertilized_until_day": -1,
                    }
                ]
            ],
        )
    )

    assert action["farmer"] == ["WATER"]
    assert action["hands"] == []
    assert action["market"] == []

    record = agent.memory.latest()

    assert record is not None
    assert record.decision_trace is not None
    assert record.decision_trace.selected_decision == record.decision_name
    assert record.decision_trace.selected_score == record.decision_score


def test_agent_memory_trace_contains_candidate_landscape():
    agent = NexusAgent()

    agent.act(
        make_observation(
            farmer=(0, 0),
            tiles=[
                [
                    {
                        "kind": "PLANT",
                        "crop": "WHEAT",
                        "planted_day": 0,
                        "watered_today": False,
                        "yield_units": 0,
                        "fertilized_until_day": -1,
                    }
                ]
            ],
        )
    )

    record = agent.memory.latest()

    assert record is not None
    assert record.decision_trace is not None

    trace = record.decision_trace

    assert trace.candidate_count >= 2
    assert len(trace.candidates) == trace.candidate_count
    assert trace.runner_up is not None
    assert trace.runner_up_score is not None


def test_agent_memory_trace_preserves_decision_margin():
    agent = NexusAgent()

    agent.act(
        make_observation(
            farmer=(0, 0),
            tiles=[
                [
                    {
                        "kind": "PLANT",
                        "crop": "WHEAT",
                        "planted_day": 0,
                        "watered_today": False,
                        "yield_units": 0,
                        "fertilized_until_day": -1,
                    }
                ]
            ],
        )
    )

    record = agent.memory.latest()

    assert record is not None
    assert record.decision_trace is not None

    trace = record.decision_trace

    assert trace.score_margin == (
        trace.selected_score - trace.runner_up_score
    )


def test_agent_trace_and_decision_remain_consistent():
    agent = NexusAgent()

    agent.act(
        make_observation(
            farmer=(0, 0),
            tiles=[
                [
                    {
                        "kind": "PLANT",
                        "crop": "WHEAT",
                        "planted_day": 0,
                        "watered_today": False,
                        "yield_units": 0,
                        "fertilized_until_day": -1,
                    }
                ]
            ],
        )
    )

    record = agent.memory.latest()

    assert record is not None
    assert record.decision_trace is not None

    trace = record.decision_trace

    assert trace.selected_decision == record.decision_name
    assert trace.selected_score == record.decision_score
    assert trace.candidate_count == len(trace.candidates)


def test_agent_plants_when_selected_target_is_current_tile():
    agent = NexusAgent()

    agent.decision_engine = FixedDecisionEngine(
        name="PLANT_WHEAT",
        action=["PLANT_WHEAT"],
    )

    observation = make_observation(
        farmer=(0, 0),
        tiles=[
            [None, None],
            [None, None],
        ],
        seeds={
            "WHEAT": 1,
        },
    )

    action = agent.act(observation)

    assert action["farmer"] == ["PLANT", "WHEAT"]

    record = agent.memory.latest()

    assert record is not None
    assert record.metadata["spatial_target"] == {
        "x": 0,
        "y": 0,
    }
    assert record.metadata["planning_target"] == {
        "x": 0,
        "y": 0,
    }
    assert record.metadata["planned_move"] == "PLANT"


def test_agent_moves_toward_nearest_plantable_tile():
    agent = NexusAgent()

    agent.decision_engine = FixedDecisionEngine(
        name="PLANT_WHEAT",
        action=["PLANT_WHEAT"],
    )

    observation = make_observation(
        farmer=(1, 1),
        tiles=[
            [
                "COOP",
                "COOP",
            ],
            [
                None,
                "COOP",
            ],
        ],
        seeds={
            "WHEAT": 1,
        },
    )

    action = agent.act(observation)

    assert action["farmer"] == ["WEST"]

    record = agent.memory.latest()

    assert record is not None
    assert record.metadata["spatial_target"] == {
        "x": 0,
        "y": 1,
    }
    assert record.metadata["planned_move"] == "WEST"


def test_agent_does_not_select_locked_or_occupied_tiles():
    agent = NexusAgent()

    agent.decision_engine = FixedDecisionEngine(
        name="PLANT_WHEAT",
        action=["PLANT_WHEAT"],
    )

    observation = make_observation(
        farmer=(0, 1),
        tiles=[
            [
                "COOP",
                "LOCKED",
            ],
            [
                {
                    "kind": "PLANT",
                    "crop": "WHEAT",
                    "planted_day": 0,
                    "watered_today": False,
                    "yield_units": 0,
                    "fertilized_until_day": -1,
                },
                None,
            ],
        ],
        seeds={
            "WHEAT": 1,
        },
    )

    action = agent.act(observation)

    assert action["farmer"] == ["EAST"]

    record = agent.memory.latest()

    assert record is not None
    assert record.metadata["spatial_target"] == {
        "x": 1,
        "y": 1,
    }


def test_agent_passes_when_no_plantable_tile_exists():
    agent = NexusAgent()

    agent.decision_engine = FixedDecisionEngine(
        name="PLANT_WHEAT",
        action=["PLANT_WHEAT"],
    )

    observation = make_observation(
        farmer=(0, 0),
        tiles=[
            [
                {
                    "kind": "PLANT",
                    "crop": "WHEAT",
                    "planted_day": 0,
                    "watered_today": False,
                    "yield_units": 0,
                    "fertilized_until_day": -1,
                },
                "LOCKED",
            ]
        ],
        seeds={
            "WHEAT": 1,
        },
    )

    action = agent.act(observation)

    assert action["farmer"] == ["PASS"]

    record = agent.memory.latest()

    assert record is not None
    assert record.metadata["spatial_target"] is None
    assert record.metadata["planning_target"] is None


def test_agent_records_spatial_target_reason():
    agent = NexusAgent()

    agent.decision_engine = FixedDecisionEngine(
        name="PLANT_WHEAT",
        action=["PLANT_WHEAT"],
    )

    observation = make_observation(
        farmer=(1, 0),
        tiles=[
            [None, "COOP"],
        ],
        seeds={
            "WHEAT": 1,
        },
    )

    action = agent.act(observation)

    assert action["farmer"] == ["WEST"]

    record = agent.memory.latest()

    assert record is not None

    reason = record.metadata["spatial_target_reason"]

    assert reason is not None
    assert "[0, 0]" in reason
    assert "1 movement steps" in reason

def test_agent_continues_plant_intent_while_navigating():
    agent = NexusAgent()

    agent.decision_engine = FixedDecisionEngine(
        name="PLANT_WHEAT",
        action=["PLANT_WHEAT"],
    )

    first_observation = make_observation(
        farmer=(1, 0),
        tiles=[
            [None, "COOP"],
        ],
        seeds={
            "WHEAT": 1,
        },
    )

    first_action = agent.act(first_observation)

    assert first_action["farmer"] == ["WEST"]
    assert agent.execution_intent is not None
    assert agent.execution_intent.is_active()
    assert (
        agent.execution_intent.decision_name
        == "PLANT_WHEAT"
    )

    second_observation = make_observation(
        farmer=(0, 0),
        tiles=[
            [None, "COOP"],
        ],
        seeds={
            "WHEAT": 1,
        },
    )
    second_observation["step"] = 2

    second_action = agent.act(second_observation)

    assert second_action["farmer"] == ["PLANT", "WHEAT"]
    assert agent.execution_intent is not None
    assert agent.execution_intent.is_active()

    assert (
        agent.execution_intent.execution_steps
        == 2
    )


def test_agent_releases_completed_plant_intent_and_requests_fresh_decision():
    class SequenceDecisionEngine:
        def __init__(self):
            self.calls = 0
            self.last_trace = None

        def decide(self, context):
            self.calls += 1

            class Decision:
                pass

            decision = Decision()

            if self.calls == 1:
                decision.name = "PLANT_WHEAT"
                decision.action = ["PLANT_WHEAT"]
                decision.score = 10.0
                decision.reason = "Plant wheat."

            else:
                decision.name = "PASS"
                decision.action = ["PASS"]
                decision.score = 1.0
                decision.reason = "No further action."

            return decision

    agent = NexusAgent()

    engine = SequenceDecisionEngine()
    agent.decision_engine = engine

    initial_observation = make_observation(
        farmer=(0, 0),
        tiles=[
            [None],
        ],
        seeds={
            "WHEAT": 1,
        },
    )

    first_action = agent.act(initial_observation)

    assert first_action["farmer"] == ["PLANT", "WHEAT"]
    assert engine.calls == 1
    assert agent.execution_intent is not None

    completed_observation = make_observation(
        farmer=(0, 0),
        tiles=[
            [
                {
                    "kind": "PLANT",
                    "crop": "WHEAT",
                    "planted_day": 1,
                    "watered_today": False,
                    "yield_units": 0,
                    "fertilized_until_day": -1,
                }
            ],
        ],
        seeds={
            "WHEAT": 0,
        },
    )
    completed_observation["step"] = 2

    action = agent.act(completed_observation)

    assert engine.calls == 2
    assert action["farmer"] == ["PASS"]

    assert agent.execution_intent is None
    assert agent.planning_target is None
    assert agent.selected_spatial_target is None

    record = agent.memory.latest()

    assert record is not None
    assert record.decision_name == "PASS"


def test_agent_releases_failed_plant_intent_and_requests_fresh_decision():
    class SequenceDecisionEngine:
        def __init__(self):
            self.calls = 0
            self.last_trace = None

        def decide(self, context):
            self.calls += 1

            class Decision:
                pass

            decision = Decision()

            if self.calls == 1:
                decision.name = "PLANT_WHEAT"
                decision.action = ["PLANT_WHEAT"]
                decision.score = 10.0
                decision.reason = "Plant wheat."

            else:
                decision.name = "PASS"
                decision.action = ["PASS"]
                decision.score = 1.0
                decision.reason = "No further action."

            return decision

    agent = NexusAgent()

    engine = SequenceDecisionEngine()
    agent.decision_engine = engine

    initial_observation = make_observation(
        farmer=(1, 0),
        tiles=[
            [None, "COOP"],
        ],
        seeds={
            "WHEAT": 1,
        },
    )

    first_action = agent.act(initial_observation)

    assert first_action["farmer"] == ["WEST"]
    assert engine.calls == 1
    assert agent.execution_intent is not None

    failed_observation = make_observation(
        farmer=(1, 0),
        tiles=[
            ["LOCKED", "COOP"],
        ],
        seeds={
            "WHEAT": 1,
        },
    )
    failed_observation["step"] = 2

    action = agent.act(failed_observation)

    assert engine.calls == 2
    assert action["farmer"] == ["PASS"]

    assert agent.execution_intent is None
    assert agent.planning_target is None
    assert agent.selected_spatial_target is None

    record = agent.memory.latest()

    assert record is not None
    assert record.decision_name == "PASS"


def test_agent_completes_clear_weed_intent_and_releases_it():
    class SequenceDecisionEngine:
        def __init__(self):
            self.calls = 0
            self.last_trace = None

        def decide(self, context):
            self.calls += 1

            class Decision:
                pass

            decision = Decision()

            if self.calls == 1:
                decision.name = "CLEAR_WEED"
                decision.action = [
                    "CLEAR_WEED",
                    0,
                    0,
                ]
                decision.score = 10.0
                decision.reason = "Clear weed."

            else:
                decision.name = "PASS"
                decision.action = ["PASS"]
                decision.score = 1.0
                decision.reason = "No further action."

            return decision

    agent = NexusAgent()

    engine = SequenceDecisionEngine()
    agent.decision_engine = engine

    initial_observation = make_observation(
        farmer=(0, 0),
        tiles=[
            [None],
        ],
    )

    initial_observation["farms"][0]["weeds"] = [
        {
            "col": 0,
            "row": 0,
        }
    ]

    first_action = agent.act(initial_observation)

    assert first_action["farmer"] == ["DIG"]
    assert engine.calls == 1
    assert agent.execution_intent is not None

    cleared_observation = make_observation(
        farmer=(0, 0),
        tiles=[
            [None],
        ],
    )
    cleared_observation["step"] = 2
    cleared_observation["farms"][0]["weeds"] = []

    action = agent.act(cleared_observation)

    assert engine.calls == 2
    assert action["farmer"] == ["PASS"]

    assert agent.execution_intent is None
    assert agent.planning_target is None
    assert agent.selected_spatial_target is None

