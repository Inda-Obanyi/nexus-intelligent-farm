from nexus.agents.nexus.runner import NexusAgentRunner


def test_nexus_runner_completes_episode():
    runner = NexusAgentRunner(
        episode_steps=24,
        player_id=0,
        debug=False,
    )

    result = runner.run()

    assert "observation" in result
    assert "decisions" in result
    assert "memory" in result

    assert len(result["decisions"]) > 0
    assert len(result["memory"]) == len(result["decisions"])

    observation = result["observation"]

    assert observation["day"] >= 0
    assert observation["hour"] >= 0
    assert "farms" in observation
    assert "private" in observation


def test_nexus_runner_records_decision_history():
    runner = NexusAgentRunner(
        episode_steps=10,
        player_id=0,
        debug=False,
    )

    result = runner.run()

    assert len(result["memory"]) == len(result["decisions"])

    latest = result["memory"][-1]

    assert latest.step >= 0
    assert latest.day >= 0
    assert latest.decision_name
    assert latest.decision_action
    assert latest.executed_action is not None


def test_nexus_runner_returns_economic_evaluation():
    runner = NexusAgentRunner(
        episode_steps=24,
        player_id=0,
        debug=False,
    )

    result = runner.run()

    assert "evaluation" in result

    evaluation = result["evaluation"]

    assert "starting_money" in evaluation
    assert "final_money" in evaluation
    assert "inventory_value" in evaluation
    assert "net_profit" in evaluation
    assert "roi_percent" in evaluation

    assert evaluation["starting_money"] == 3000.0
    assert evaluation["final_money"] >= 0
