from nexus.environments.kaggriculture.environment import KaggricultureEnvironment


def test_market_purchase_is_not_available_to_farmer_in_same_step():
    env = KaggricultureEnvironment(
        episode_steps=10,
        debug=True,
    )
    env.reset()

    observation = env.state[0].observation
    farm = observation["farms"][0]

    # Initial farmer position is [4, 4].
    x, y = farm["farmer"]
    assert [x, y] == [4, 4]

    # Attempt to buy a wheat seed and plant it in the SAME step.
    env.step(
        [
            {
                "farmer": ["PLANT", "WHEAT"],
                "hands": [],
                "market": [["BUY_SEED", "WHEAT", 1]],
            },
            {
                "farmer": ["PASS"],
                "hands": [],
                "market": [],
            },
        ]
    )

    observation = env.state[0].observation
    tile = observation["farms"][0]["tiles"][y][x]

    # The seed purchase occurs, but the farmer cannot use
    # that newly purchased seed until a later simulation step.
    assert tile is None

    # Confirm the seed is now available after the step.
    assert observation["private"]["seeds"].get("WHEAT", 0) == 1


def test_purchased_seed_can_be_used_on_following_step():
    env = KaggricultureEnvironment(
        episode_steps=10,
        debug=True,
    )
    env.reset()

    # Step 0: purchase seed.
    env.step(
        [
            {
                "farmer": ["PASS"],
                "hands": [],
                "market": [["BUY_SEED", "WHEAT", 1]],
            },
            {
                "farmer": ["PASS"],
                "hands": [],
                "market": [],
            },
        ]
    )

    # Step 1: use the purchased seed.
    env.step(
        [
            {
                "farmer": ["PLANT", "WHEAT"],
                "hands": [],
                "market": [],
            },
            {
                "farmer": ["PASS"],
                "hands": [],
                "market": [],
            },
        ]
    )

    observation = env.state[0].observation
    farm = observation["farms"][0]
    x, y = farm["farmer"]
    tile = farm["tiles"][y][x]

    assert tile is not None
    assert tile["kind"] == "PLANT"
    assert tile["crop"] == "WHEAT"
