from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class PlanningTarget:
    """
    Represents a farm location that the planner should reach.

    Coordinates follow the Kaggriculture [x, y] convention.
    """

    x: int
    y: int

    @property
    def position(self) -> List[int]:
        """Return the target using the environment's [x, y] convention."""
        return [self.x, self.y]


class FarmPlanner:
    """
    Lightweight goal-directed planner for farm navigation.

    The planner produces exactly one movement action at a time.
    It does not execute actions and does not interact with the
    environment directly.

    Current routing strategy:
        - Move horizontally until x matches the target.
        - Move vertically until y matches the target.
        - Return None when the target has been reached.
    """

    def next_move(
        self,
        current_position: List[int],
        target: PlanningTarget,
    ) -> Optional[str]:
        """
        Return the next movement required to reach the target.

        Parameters
        ----------
        current_position:
            Farmer position using [x, y].

        target:
            Desired farm location.

        Returns
        -------
        Optional[str]
            One of:
                "WEST"
                "EAST"
                "NORTH"
                "SOUTH"

            Returns None when the farmer is already at the target.
        """

        current_x, current_y = current_position

        if current_x < target.x:
            return "EAST"

        if current_x > target.x:
            return "WEST"

        if current_y < target.y:
            return "SOUTH"

        if current_y > target.y:
            return "NORTH"

        return None
