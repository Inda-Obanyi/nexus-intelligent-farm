from nexus.evaluation.metrics import (
    calculate_inventory_value,
    calculate_net_profit,
    calculate_roi,
    evaluate_episode,
)


def test_calculate_inventory_value():
    shed = {
        "WHEAT": 2,
        "CARROT": 3,
        "MILK": 1,
    }

    expected = (
        2 * 25
        + 3 * 35
        + 1 * 160
    )

    assert calculate_inventory_value(shed) == expected


def test_calculate_inventory_value_ignores_unknown_items():
    shed = {
        "WHEAT": 2,
        "UNKNOWN_ITEM": 100,
    }

    assert calculate_inventory_value(shed) == 50


def test_calculate_net_profit_includes_inventory_value():
    profit = calculate_net_profit(
        starting_money=3000,
        final_money=3200,
        shed={
            "WHEAT": 4,
        },
    )

    assert profit == 300


def test_calculate_roi():
    roi = calculate_roi(
        starting_money=3000,
        net_profit=150,
    )

    assert roi == 5.0


def test_calculate_roi_zero_starting_money():
    roi = calculate_roi(
        starting_money=0,
        net_profit=500,
    )

    assert roi == 0.0


def test_evaluate_episode_returns_expected_metrics():
    result = evaluate_episode(
        starting_money=3000,
        final_money=3140,
        shed={},
    )

    assert result["starting_money"] == 3000
    assert result["final_money"] == 3140
    assert result["inventory_value"] == 0
    assert result["net_profit"] == 140

    assert abs(
        result["roi_percent"] - 4.6666666667
    ) < 1e-9


def test_evaluate_episode_includes_remaining_inventory():
    result = evaluate_episode(
        starting_money=3000,
        final_money=3000,
        shed={
            "WHEAT": 2,
            "CARROT": 1,
        },
    )

    # 2 WHEAT = 50
    # 1 CARROT = 35
    # Total inventory = 85
    assert result["inventory_value"] == 85
    assert result["net_profit"] == 85

    assert abs(
        result["roi_percent"]
        - (85 / 3000 * 100)
    ) < 1e-9


def test_evaluate_episode_uses_roi_percent_contract():
    result = evaluate_episode(
        starting_money=1000,
        final_money=1100,
        shed={},
    )

    assert "roi_percent" in result
    assert "roi" not in result
    assert result["roi_percent"] == 10.0
