from typing import Dict


MARKET_PRICES = {
    "WHEAT": 25,
    "CARROT": 35,
    "TOMATO": 60,
    "STRAWBERRY": 120,
    "MELON": 250,
    "EGG": 50,
    "MILK": 160,
    "WOOL": 200,
    "FERTILIZER": 100,
}


def calculate_inventory_value(shed: Dict[str, int]) -> float:
    """Calculate the market value of all products/resources in the shed."""

    value = 0.0

    for item, quantity in shed.items():
        price = MARKET_PRICES.get(item, 0)
        value += quantity * price

    return value


def calculate_net_profit(
    starting_money: float,
    final_money: float,
    shed: Dict[str, int],
) -> float:
    """
    Calculate economic profit including the value of remaining inventory.

    Net Profit =
        Final Cash
        + Remaining Inventory Value
        - Starting Cash
    """

    inventory_value = calculate_inventory_value(shed)

    return (
        final_money
        + inventory_value
        - starting_money
    )


def calculate_roi(
    starting_money: float,
    net_profit: float,
) -> float:
    """Calculate return on investment as a percentage."""

    if starting_money == 0:
        return 0.0

    return (net_profit / starting_money) * 100


def evaluate_episode(
    starting_money: float,
    final_money: float,
    shed: Dict[str, int],
) -> Dict[str, float]:
    """Return the main economic metrics for an episode."""

    inventory_value = calculate_inventory_value(shed)

    net_profit = calculate_net_profit(
        starting_money=starting_money,
        final_money=final_money,
        shed=shed,
    )

    roi = calculate_roi(
        starting_money=starting_money,
        net_profit=net_profit,
    )

    return {
        "starting_money": starting_money,
        "final_money": final_money,
        "inventory_value": inventory_value,
        "net_profit": net_profit,
        "roi_percent": roi,
    }
