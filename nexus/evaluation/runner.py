from nexus.environments.kaggriculture.environment import KaggricultureEnvironment
from nexus.agents.baseline.agent import BaselineAgent
from nexus.evaluation.metrics import evaluate_episode


STARTING_MONEY = 3000.0
EPISODE_STEPS = 720


def main():
    env = KaggricultureEnvironment(
        episode_steps=EPISODE_STEPS,
        debug=True,
    )

    env.reset()

    agent = BaselineAgent(player_id=0)

    opponent_action = {
        "farmer": ["PASS"],
        "hands": [],
        "market": [],
    }

    print("=" * 70)
    print("NEXUS BASELINE V1 — ECONOMIC EVALUATION")
    print("=" * 70)

    for step in range(EPISODE_STEPS):
        if env.state[0].status != "ACTIVE":
            break

        observation = env.state[0].observation
        action = agent.act(observation)

        env.step([
            action,
            opponent_action,
        ])

    observation = env.state[0].observation

    final_money = observation["farms"][0]["money"]
    shed = observation["private"]["shed"]

    report = evaluate_episode(
        starting_money=STARTING_MONEY,
        final_money=final_money,
        shed=shed,
    )

    print()
    print("-" * 70)
    print("EPISODE RESULTS")
    print("-" * 70)
    print(f"Final Day:          {observation['day']}")
    print(f"Final Hour:         {observation['hour']}")
    print(f"Final Cash:         {report['final_money']:.2f}")
    print(f"Inventory Value:    {report['inventory_value']:.2f}")
    print(f"Net Profit:         {report['net_profit']:.2f}")
    print(f"ROI:                {report['roi_percent']:.2f}%")
    print()
    print("Remaining Inventory:")
    for item, quantity in shed.items():
        if quantity > 0:
            print(f"  {item}: {quantity}")

    print("-" * 70)
    print("NEXUS BASELINE V1 COMPLETE")
    print("-" * 70)


if __name__ == "__main__":
    main()
