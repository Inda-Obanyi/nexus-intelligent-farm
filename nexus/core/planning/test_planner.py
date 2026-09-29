from nexus.core.planning.planner import (
    FarmPlanner,
    PlanningTarget,
)


def test_planner_moves_west_when_target_is_left():
    planner = FarmPlanner()
    target = PlanningTarget(2, 1)

    assert planner.next_move(
        [4, 4],
        target,
    ) == "WEST"


def test_planner_moves_east_when_target_is_right():
    planner = FarmPlanner()
    target = PlanningTarget(6, 4)

    assert planner.next_move(
        [4, 4],
        target,
    ) == "EAST"


def test_planner_moves_north_when_target_is_above():
    planner = FarmPlanner()
    target = PlanningTarget(4, 1)

    assert planner.next_move(
        [4, 4],
        target,
    ) == "NORTH"


def test_planner_moves_south_when_target_is_below():
    planner = FarmPlanner()
    target = PlanningTarget(4, 7)

    assert planner.next_move(
        [4, 4],
        target,
    ) == "SOUTH"


def test_planner_returns_none_when_target_reached():
    planner = FarmPlanner()
    target = PlanningTarget(4, 4)

    assert planner.next_move(
        [4, 4],
        target,
    ) is None


def test_planner_reaches_target_using_one_move_at_a_time():
    planner = FarmPlanner()
    target = PlanningTarget(2, 1)

    position = [4, 4]

    expected_moves = [
        "WEST",
        "WEST",
        "NORTH",
        "NORTH",
        "NORTH",
    ]

    actual_moves = []

    for _ in expected_moves:
        move = planner.next_move(
            position,
            target,
        )

        actual_moves.append(move)

        if move == "WEST":
            position[0] -= 1
        elif move == "EAST":
            position[0] += 1
        elif move == "NORTH":
            position[1] -= 1
        elif move == "SOUTH":
            position[1] += 1

    assert actual_moves == expected_moves
    assert position == target.position
