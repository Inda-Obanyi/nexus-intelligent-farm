from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CropState:
    crop: str
    row: int
    col: int
    planted_day: int
    watered_today: bool
    yield_units: int
    fertilized_until_day: int


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
    crops: List[CropState] = field(default_factory=list)
    resources: FarmResources = field(
        default_factory=lambda: FarmResources(money=0.0)
    )
    market: MarketState = field(
        default_factory=MarketState
    )
    raw_observation: Optional[Dict[str, Any]] = None


class NexusStateBuilder:
    """Convert an environment observation into a NEXUS-native state."""

    def __init__(self, player_id: int = 0):
        self.player_id = player_id

    def build(self, observation: Dict[str, Any]) -> FarmState:
        farm = observation["farms"][self.player_id]
        private = observation["private"]

        crops = []

        for row_index, row in enumerate(farm["tiles"]):
            for col_index, tile in enumerate(row):
                if not isinstance(tile, dict):
                    continue

                if tile.get("kind") != "PLANT":
                    continue

                crops.append(
                    CropState(
                        crop=tile.get("crop", "UNKNOWN"),
                        row=row_index,
                        col=col_index,
                        planted_day=tile.get("planted_day", 0),
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
                    )
                )

        resources = FarmResources(
            money=farm["money"],
            seeds=dict(private["seeds"]),
            shed=dict(private["shed"]),
        )

        market_observation = observation.get(
            "market",
            {}
        )

        market = MarketState(
            prices=dict(
                market_observation.get(
                    "prices",
                    {}
                )
            ),
            inventory=dict(
                market_observation.get(
                    "inventory",
                    {}
                )
            ),
        )

        return FarmState(
            player_id=self.player_id,
            day=observation["day"],
            hour=observation["hour"],
            farmer_position=list(farm["farmer"]),
            crops=crops,
            resources=resources,
            market=market,
            raw_observation=observation,
        )
