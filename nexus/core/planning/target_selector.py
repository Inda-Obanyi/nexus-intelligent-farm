from dataclasses import dataclass
from typing import Optional

from nexus.core.state import FarmState, FarmTileState
from nexus.intelligence.crops.definitions import get_crop_definition


@dataclass(frozen=True)
class SpatialTarget:
    """
    A spatial target selected from the observable farm state.

    Coordinates use the Kaggriculture convention:

        position = [x, y]
    """

    x: int
    y: int
    reason: str

    @property
    def position(self) -> list[int]:
        return [self.x, self.y]

    @property
    def distance_from_origin(self) -> int:
        """
        Manhattan distance from [0, 0].

        This is descriptive only and is not used as a
        decision-quality metric.
        """
        return abs(self.x) + abs(self.y)


class SpatialTargetSelector:
    """
    Selects spatial targets from FarmState.

    This layer does not decide what the farm should produce.
    It only answers spatial questions such as:

        "Where can this objective happen?"

    Strategic decisions remain in the DecisionEngine.
    Navigation remains in FarmPlanner.
    """

    @staticmethod
    def manhattan_distance(
        origin: list[int],
        target: list[int],
    ) -> int:
        """
        Calculate Manhattan distance between two [x, y]
        coordinates.
        """

        return (
            abs(target[0] - origin[0])
            + abs(target[1] - origin[1])
        )

    def nearest_empty_tile(
        self,
        state: FarmState,
    ) -> Optional[SpatialTarget]:
        """
        Select the nearest currently empty usable tile.

        Returns None when no empty usable tile exists.

        Selection is deterministic:
        1. shortest Manhattan distance from farmer
        2. lowest y coordinate
        3. lowest x coordinate
        """

        farmer_position = list(
            state.farmer_position
        )

        candidates = state.empty_tiles()

        if not candidates:
            return None

        target = min(
            candidates,
            key=lambda tile: (
                self.manhattan_distance(
                    farmer_position,
                    tile.position,
                ),
                tile.row,
                tile.col,
            ),
        )

        distance = self.manhattan_distance(
            farmer_position,
            target.position,
        )

        return SpatialTarget(
            x=target.col,
            y=target.row,
            reason=(
                f"Nearest empty usable tile is "
                f"[{target.col}, {target.row}], "
                f"{distance} movement steps from "
                f"the farmer at "
                f"{farmer_position}."
            ),
        )

    def nearest_plantable_tile(
        self,
        state: FarmState,
    ) -> Optional[SpatialTarget]:
        """
        Select the nearest currently plantable tile.

        Selection is deterministic:
        1. shortest Manhattan distance from farmer
        2. lowest y coordinate
        3. lowest x coordinate
        """

        farmer_position = list(
            state.farmer_position
        )

        candidates = state.plantable_tiles()

        if not candidates:
            return None

        target = min(
            candidates,
            key=lambda tile: (
                self.manhattan_distance(
                    farmer_position,
                    tile.position,
                ),
                tile.row,
                tile.col,
            ),
        )

        distance = self.manhattan_distance(
            farmer_position,
            target.position,
        )

        return SpatialTarget(
            x=target.col,
            y=target.row,
            reason=(
                f"Nearest plantable tile is "
                f"[{target.col}, {target.row}], "
                f"{distance} movement steps from "
                f"the farmer at "
                f"{farmer_position}."
            ),
        )

    def nearest_harvestable_crop(
        self,
        state: FarmState,
    ) -> Optional[SpatialTarget]:
        """
        Select the nearest harvestable crop on the farm.

        A crop is harvestable when:
        - it has positive yield units
        - it has reached its crop-specific first-yield day

        Selection is deterministic:
        1. shortest Manhattan distance from farmer
        2. lowest y coordinate
        3. lowest x coordinate
        """

        farmer_position = list(
            state.farmer_position
        )

        candidates = []

        for crop in state.crops:
            if crop.yield_units <= 0:
                continue

            crop_definition = get_crop_definition(
                crop.crop
            )

            crop_age_days = (
                state.day - crop.planted_day
            )

            if (
                crop_age_days
                < crop_definition.first_yield_day
            ):
                continue

            candidates.append(crop)

        if not candidates:
            return None

        target_crop = min(
            candidates,
            key=lambda crop: (
                self.manhattan_distance(
                    farmer_position,
                    [crop.col, crop.row],
                ),
                crop.row,
                crop.col,
            ),
        )

        distance = self.manhattan_distance(
            farmer_position,
            [target_crop.col, target_crop.row],
        )

        return SpatialTarget(
            x=target_crop.col,
            y=target_crop.row,
            reason=(
                f"Nearest harvestable "
                f"{target_crop.crop} crop is "
                f"[{target_crop.col}, {target_crop.row}], "
                f"{distance} movement steps from "
                f"the farmer at "
                f"{farmer_position}."
            ),
        )

    def target_for_tile(
        self,
        tile: FarmTileState,
        state: FarmState,
    ) -> Optional[SpatialTarget]:
        """
        Convert a validated FarmTileState into a planning target.

        Locked or occupied tiles cannot become spatial targets.
        """

        if tile.locked:
            return None

        if tile.occupied:
            return None

        distance = self.manhattan_distance(
            state.farmer_position,
            tile.position,
        )

        return SpatialTarget(
            x=tile.col,
            y=tile.row,
            reason=(
                f"Selected usable tile "
                f"[{tile.col}, {tile.row}], "
                f"{distance} movement steps from "
                f"the farmer."
            ),
        )
