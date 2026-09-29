from nexus.core.state import NexusStateBuilder


def make_observation():
    tiles = [[None for _ in range(10)] for _ in range(10)]

    # Simulator tile coordinates:
    # x = 7, y = 2
    tiles[2][7] = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 0,
        "watered_today": True,
        "yield_units": 1,
        "fertilized_until_day": -1,
    }

    return {
        "day": 2,
        "hour": 0,
        "farms": [
            {
                "farmer": [7, 3],
                "money": 3000.0,
                "tiles": tiles,
            }
        ],
        "private": {
            "seeds": {},
            "shed": {},
        },
        "market": {
            "prices": {},
            "inventory": {},
        },
    }


def test_farmer_position_preserves_x_y_order():
    state = NexusStateBuilder(player_id=0).build(make_observation())

    assert state.farmer_position == [7, 3]


def test_crop_tile_coordinates_map_row_y_and_col_x():
    state = NexusStateBuilder(player_id=0).build(make_observation())

    assert len(state.crops) == 1

    crop = state.crops[0]

    assert crop.row == 2
    assert crop.col == 7
