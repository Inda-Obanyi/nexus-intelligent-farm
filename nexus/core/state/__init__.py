from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nexus.intelligence.crops.definitions import get_crop_definition


@dataclass
class CropState:
    crop: str
    row: int
    col: int
    planted_day: int
    watered_today: bool
    yield_units: int
    fertilized_until_day: int
    ongoing: bool = False

    @property
    def harvestable(self) -> bool:
        return self.yield_units > 0


@dataclass(frozen=True)
class WeedState:
    row: int
    col: int

    @property
    def position(self) -> List[int]:
        return [self.col, self.row]


@dataclass(frozen=True)
class FarmTileState:
    """
    Observable state of one farm tile.

    Coordinates follow the Kaggriculture convention:

        position = [x, y]

    Internally the observation is stored as:

        tiles[y][x]
    """

    row: int
    col: int
    kind: str
    locked: bool
    occupied: bool

    @property
    def position(self) -> List[int]:
        """Return the tile position as [x, y]."""
        return [self.col, self.row]

    @property
    def empty(self) -> bool:
        """
        Return True when the tile is unlocked and empty.
        """
        return not self.locked and not self.occupied

    @property
    def plantable(self) -> bool:
        """
        Return whether the tile is currently available
        as an empty usable farm tile.

        This is a spatial observation, not an agricultural
        recommendation.
        """
        return self.empty


@dataclass
class FarmResources:
    money: float
    seeds: Dict[str, int] = field(default_factory=dict)
    shed: Dict[str, int] = field(default_factory=dict)


@dataclass
class MarketState:
    prices: Dict[str, float] = field(default_factory=dict)
    inventory: Dict[str, int] = field(default_factory=dict)


@dataclass
class FarmState:
    player_id: int
    day: int
    hour: int
    farmer_position: List[int]

    tiles: List[FarmTileState] = field(
        default_factory=list
    )

    crops: List[CropState] = field(
        default_factory=list
    )

    weeds: List[WeedState] = field(
        default_factory=list
    )

    resources: FarmResources = field(
        default_factory=lambda: FarmResources(
            money=0.0
        )
    )

    market: MarketState = field(
        default_factory=MarketState
    )

    raw_observation: Optional[
        Dict[str, Any]
    ] = None

    def empty_tiles(self) -> List[FarmTileState]:
        """
        Return all unlocked empty farm tiles.
        """
        return [
            tile
            for tile in self.tiles
            if tile.empty
        ]

    def plantable_tiles(self) -> List[FarmTileState]:
        """
        Return all currently plantable spatial tiles.

        This is equivalent to empty unlocked tiles at the
        current simulation layer.
        """
        return [
            tile
            for tile in self.tiles
            if tile.plantable
        ]

    def tile_at(
        self,
        x: int,
        y: int,
    ) -> Optional[FarmTileState]:
        """
        Return the observable tile at [x, y].

        Returns None when the coordinates are outside the
        represented board.
        """
        for tile in self.tiles:
            if tile.col == x and tile.row == y:
                return tile

        return None


class NexusStateBuilder:
    def __init__(
        self,
        player_id: int = 0,
    ):
        self.player_id = player_id

    def build(
        self,
        observation: Dict[str, Any],
    ) -> FarmState:
        farm = observation["farms"][
            self.player_id
        ]

        private = observation["private"]

        crops = []
        weeds = []
        tiles = []

        for row_index, row in enumerate(
            farm["tiles"]
        ):
            for col_index, tile in enumerate(
                row
            ):
                if tile == "LOCKED":
                    tiles.append(
                        FarmTileState(
                            row=row_index,
                            col=col_index,
                            kind="LOCKED",
                            locked=True,
                            occupied=False,
                        )
                    )

                    continue

                if tile is None:
                    tiles.append(
                        FarmTileState(
                            row=row_index,
                            col=col_index,
                            kind="EMPTY",
                            locked=False,
                            occupied=False,
                        )
                    )

                    continue

                if not isinstance(tile, dict):
                    tiles.append(
                        FarmTileState(
                            row=row_index,
                            col=col_index,
                            kind=str(tile),
                            locked=False,
                            occupied=True,
                        )
                    )

                    continue

                kind = tile.get(
                    "kind",
                    "UNKNOWN",
                )

                tiles.append(
                    FarmTileState(
                        row=row_index,
                        col=col_index,
                        kind=kind,
                        locked=False,
                        occupied=True,
                    )
                )

                if kind == "WEED":
                    weeds.append(
                        WeedState(
                            row=row_index,
                            col=col_index,
                        )
                    )

                    continue

                if kind != "PLANT":
                    continue

                crop_name = tile.get(
                    "crop",
                    "UNKNOWN",
                )

                crops.append(
                    CropState(
                        crop=crop_name,
                        row=row_index,
                        col=col_index,
                        planted_day=tile.get(
                            "planted_day",
                            0,
                        ),
                        watered_today=tile.get(
                            "watered_today",
                            False,
                        ),
                        yield_units=tile.get(
                            "yield_units",
                            0,
                        ),
                        fertilized_until_day=tile.get(
                            "fertilized_until_day",
                            -1,
                        ),
                        ongoing=get_crop_definition(
                            crop_name
                        ).ongoing,
                    )
                )

        resources = FarmResources(
            money=farm["money"],
            seeds=dict(
                private["seeds"]
            ),
            shed=dict(
                private["shed"]
            ),
        )

        market_observation = observation.get(
            "market",
            {},
        )

        market = MarketState(
            prices=dict(
                market_observation.get(
                    "prices",
                    {},
                )
            ),
            inventory=dict(
                market_observation.get(
                    "inventory",
                    {},
                )
            ),
        )

        return FarmState(
            player_id=self.player_id,
            day=observation["day"],
            hour=observation["hour"],
            farmer_position=list(
                farm["farmer"]
            ),
            tiles=tiles,
            crops=crops,
            weeds=weeds,
            resources=resources,
            market=market,
            raw_observation=observation,
        )
