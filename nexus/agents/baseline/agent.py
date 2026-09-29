class BaselineAgent:
    """Deterministic wheat farming baseline for Kaggriculture."""

    def __init__(self, player_id=0):
        self.player_id = player_id

        self.route = [
            [4, 4], [3, 4], [2, 4], [1, 4], [0, 4],
            [0, 3], [1, 3], [2, 3], [3, 3], [4, 3],
            [4, 2], [3, 2], [2, 2], [1, 2], [0, 2],
            [0, 1], [1, 1], [2, 1], [3, 1], [4, 1],
            [4, 0], [3, 0], [2, 0], [1, 0], [0, 0],
        ]

    def _movement_action(self, current, target):
        current_row, current_col = current
        target_row, target_col = target

        if current_row < target_row:
            return ["EAST"]

        if current_row > target_row:
            return ["WEST"]

        if current_col < target_col:
            return ["SOUTH"]

        if current_col > target_col:
            return ["NORTH"]

        return None

    def act(self, observation):
        farm = observation["farms"][self.player_id]
        private = observation["private"]

        current = farm["farmer"]
        row, col = current
        tile = farm["tiles"][row][col]

        seeds = private["seeds"]
        wheat_seeds = seeds.get("WHEAT", 0)

        # 1. Buy wheat seed if inventory is empty.
        if wheat_seeds == 0:
            return {
                "farmer": ["PASS"],
                "hands": [],
                "market": [
                    ["BUY_SEED", "WHEAT", 1]
                ],
            }

        # 2. Manage an existing wheat plant.
        if isinstance(tile, dict):
            if tile.get("kind") == "PLANT":
                if tile.get("crop") == "WHEAT":

                    # Harvest mature wheat.
                    if (
                        observation["day"]
                        >= tile.get("planted_day", 0) + 2
                        and tile.get("yield_units", 0) > 0
                    ):
                        return {
                            "farmer": ["HARVEST"],
                            "hands": [],
                            "market": [],
                        }

                    # Water wheat if necessary.
                    if not tile.get("watered_today", False):
                        return {
                            "farmer": ["WATER"],
                            "hands": [],
                            "market": [],
                        }

        # 3. Plant wheat on an empty tile.
        if tile is None and wheat_seeds > 0:
            return {
                "farmer": ["PLANT", "WHEAT"],
                "hands": [],
                "market": [],
            }

        # 4. Navigate through the farm.
        try:
            current_index = self.route.index(current)
        except ValueError:
            current_index = -1

        if current_index + 1 < len(self.route):
            target = self.route[current_index + 1]

            movement = self._movement_action(
                current,
                target,
            )

            if movement:
                return {
                    "farmer": movement,
                    "hands": [],
                    "market": [],
                }

        # 5. Nothing else to do.
        return {
            "farmer": ["PASS"],
            "hands": [],
            "market": [],
        }
