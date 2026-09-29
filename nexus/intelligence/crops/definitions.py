from dataclasses import dataclass


@dataclass(frozen=True)
class CropDefinition:
    name: str
    seed_cost: float
    first_yield_day: int
    max_yield_day: int
    max_yield: int
    harvest_interval: int
    ongoing: bool


CROP_DEFINITIONS = {
    "WHEAT": CropDefinition(
        name="WHEAT",
        seed_cost=10,
        first_yield_day=2,
        max_yield_day=4,
        max_yield=6,
        harvest_interval=0,
        ongoing=False,
    ),
    "CARROT": CropDefinition(
        name="CARROT",
        seed_cost=20,
        first_yield_day=2,
        max_yield_day=3,
        max_yield=4,
        harvest_interval=0,
        ongoing=False,
    ),
    "TOMATO": CropDefinition(
        name="TOMATO",
        seed_cost=50,
        first_yield_day=8,
        max_yield_day=8,
        max_yield=4,
        harvest_interval=1,
        ongoing=True,
    ),
    "STRAWBERRY": CropDefinition(
        name="STRAWBERRY",
        seed_cost=100,
        first_yield_day=10,
        max_yield_day=10,
        max_yield=4,
        harvest_interval=2,
        ongoing=True,
    ),
    "MELON": CropDefinition(
        name="MELON",
        seed_cost=80,
        first_yield_day=10,
        max_yield_day=12,
        max_yield=6,
        harvest_interval=0,
        ongoing=False,
    ),
}


def get_crop_definition(crop_name: str) -> CropDefinition:
    try:
        return CROP_DEFINITIONS[crop_name.upper()]
    except KeyError as exc:
        raise ValueError(f"Unknown crop: {crop_name}") from exc
