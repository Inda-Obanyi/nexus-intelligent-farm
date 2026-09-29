from nexus.core.planning.target_selector import (
    SpatialTargetSelector,
)
from nexus.core.state import (
    FarmResources,
    FarmState,
    FarmTileState,
    MarketState,
)


def build_state(
    farmer_position,
    tiles,
):
    return FarmState(
        player_id=0,
        day=1,
        hour=0,
        farmer_position=list(farmer_position),
        tiles=tiles,
        crops=[],
        weeds=[],
        resources=FarmResources(money=3000),
        market=MarketState(),
    )


def make_tile(
    x,
    y,
    kind="EMPTY",
    locked=False,
    occupied=False,
):
    return FarmTileState(
        row=y,
        col=x,
        kind=kind,
        locked=locked,
        occupied=occupied,
    )


def test_nearest_empty_tile_can_be_current_farmer_tile():
    state = build_state(
        farmer_position=[4, 4],
        tiles=[
            make_tile(4, 4),
            make_tile(3, 4),
            make_tile(0, 0),
        ],
    )

    selector = SpatialTargetSelector()

    target = selector.nearest_empty_tile(state)

    assert target is not None
    assert target.position == [4, 4]


def test_nearest_plantable_tile_can_be_current_farmer_tile():
    state = build_state(
        farmer_position=[4, 4],
        tiles=[
            make_tile(4, 4),
            make_tile(2, 4),
            make_tile(0, 0),
        ],
    )

    selector = SpatialTargetSelector()

    target = selector.nearest_plantable_tile(state)

    assert target is not None
    assert target.position == [4, 4]


def test_locked_tiles_are_not_empty_targets():
    state = build_state(
        farmer_position=[4, 4],
        tiles=[
            make_tile(
                3,
                4,
                kind="LOCKED",
                locked=True,
            ),
            make_tile(2, 4),
        ],
    )

    selector = SpatialTargetSelector()

    target = selector.nearest_empty_tile(state)

    assert target is not None
    assert target.position == [2, 4]


def test_occupied_tiles_are_not_empty_targets():
    state = build_state(
        farmer_position=[4, 4],
        tiles=[
            make_tile(
                3,
                4,
                kind="PLANT",
                occupied=True,
            ),
            make_tile(2, 4),
        ],
    )

    selector = SpatialTargetSelector()

    target = selector.nearest_empty_tile(state)

    assert target is not None
    assert target.position == [2, 4]


def test_no_empty_tiles_returns_none():
    state = build_state(
        farmer_position=[4, 4],
        tiles=[
            make_tile(
                3,
                4,
                kind="PLANT",
                occupied=True,
            ),
            make_tile(
                2,
                4,
                kind="LOCKED",
                locked=True,
            ),
        ],
    )

    selector = SpatialTargetSelector()

    assert selector.nearest_empty_tile(state) is None


def test_target_for_valid_empty_tile():
    state = build_state(
        farmer_position=[4, 4],
        tiles=[
            make_tile(3, 4),
        ],
    )

    selector = SpatialTargetSelector()

    target = selector.target_for_tile(
        state.tiles[0],
        state,
    )

    assert target is not None
    assert target.position == [3, 4]


def test_target_for_locked_tile_returns_none():
    state = build_state(
        farmer_position=[4, 4],
        tiles=[
            make_tile(
                3,
                4,
                kind="LOCKED",
                locked=True,
            ),
        ],
    )

    selector = SpatialTargetSelector()

    assert (
        selector.target_for_tile(
            state.tiles[0],
            state,
        )
        is None
    )


def test_target_for_occupied_tile_returns_none():
    state = build_state(
        farmer_position=[4, 4],
        tiles=[
            make_tile(
                3,
                4,
                kind="PLANT",
                occupied=True,
            ),
        ],
    )

    selector = SpatialTargetSelector()

    assert (
        selector.target_for_tile(
            state.tiles[0],
            state,
        )
        is None
    )


def test_distance_uses_manhattan_geometry():
    selector = SpatialTargetSelector()

    distance = selector.manhattan_distance(
        [4, 4],
        [1, 2],
    )

    assert distance == 5
