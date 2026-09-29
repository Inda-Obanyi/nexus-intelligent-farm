from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class OutcomeEvaluation:
    """
    Action-aware and causally aware evaluation of a NEXUS decision.

    The evaluator separates:

        1. What NEXUS expected to happen.
        2. What the environment actually changed.
        3. Which changes can reasonably be attributed to the action.
        4. Which changes are environmental/lifecycle effects.
        5. Whether the expected effect was observed.

    This is descriptive evaluation.
    It does not claim that a decision was globally optimal.
    """

    decision_name: str
    executed: bool

    execution_effect: str

    money_delta: float
    economic_effect: str

    position_changed: bool
    crop_count_delta: int
    weed_count_delta: int

    operational_effect: str

    expected_effect: str
    observed_effect: str

    agent_effect: str
    environment_effect: str

    effect_match: str

    evidence: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "decision_name": self.decision_name,
            "executed": self.executed,
            "execution_effect": self.execution_effect,
            "money_delta": self.money_delta,
            "economic_effect": self.economic_effect,
            "position_changed": self.position_changed,
            "crop_count_delta": self.crop_count_delta,
            "weed_count_delta": self.weed_count_delta,
            "operational_effect": self.operational_effect,
            "expected_effect": self.expected_effect,
            "observed_effect": self.observed_effect,
            "agent_effect": self.agent_effect,
            "environment_effect": self.environment_effect,
            "effect_match": self.effect_match,
            "evidence": self.evidence,
        }


class DecisionOutcomeEvaluator:
    """
    Evaluate the observable consequence of a NEXUS decision.

    Decision
        ↓
    Expected Effect
        ↓
    Before State
        ↓
    Executed Action
        ↓
    After State
        ↓
    ┌──────────────────────┐
    │ Action Attribution   │
    └──────────────────────┘
        +
    ┌──────────────────────┐
    │ Environment Effects  │
    └──────────────────────┘
        ↓
    Effect Match

    Multi-step spatial decisions are evaluated in stages.

    Example:

        PLANT_WHEAT
            ↓
        EAST
            ↓
        PARTIAL
            ↓
        PLANT WHEAT
            ↓
        MATCHED
    """

    def evaluate(
        self,
        decision_name: str,
        outcome: Dict[str, Any],
        executed_action: Dict[str, Any] | None = None,
    ) -> OutcomeEvaluation:

        if executed_action is not None:
            outcome = dict(outcome)
            outcome["executed_action"] = executed_action

        executed = self._executed(outcome)

        money_before = float(
            outcome.get("money_before", 0.0)
        )
        money_after = float(
            outcome.get("money_after", 0.0)
        )

        money_delta = money_after - money_before

        position_before = outcome.get(
            "position_before"
        )
        position_after = outcome.get(
            "position_after"
        )

        position_changed = (
            position_before != position_after
        )

        crop_count_delta = (
            outcome.get("crop_count_after", 0)
            - outcome.get("crop_count_before", 0)
        )

        weed_count_delta = (
            outcome.get("weed_count_after", 0)
            - outcome.get("weed_count_before", 0)
        )

        execution_effect = self._execution_effect(
            executed
        )

        economic_effect = self._economic_effect(
            money_delta
        )

        operational_effect = self._operational_effect(
            position_changed=position_changed,
            crop_count_delta=crop_count_delta,
            weed_count_delta=weed_count_delta,
        )

        expected_effect = self._expected_effect(
            decision_name
        )

        observed_effect = self._observed_effect(
            outcome=outcome,
            money_delta=money_delta,
            position_changed=position_changed,
            crop_count_delta=crop_count_delta,
            weed_count_delta=weed_count_delta,
        )

        agent_effect = self._agent_effect(
            decision_name=decision_name,
            outcome=outcome,
            money_delta=money_delta,
            position_changed=position_changed,
            crop_count_delta=crop_count_delta,
            weed_count_delta=weed_count_delta,
        )

        environment_effect = self._environment_effect(
            decision_name=decision_name,
            outcome=outcome,
        )

        effect_match = self._effect_match(
            decision_name=decision_name,
            outcome=outcome,
            executed=executed,
            money_delta=money_delta,
            position_changed=position_changed,
            crop_count_delta=crop_count_delta,
            weed_count_delta=weed_count_delta,
        )

        return OutcomeEvaluation(
            decision_name=decision_name,
            executed=executed,
            execution_effect=execution_effect,
            money_delta=money_delta,
            economic_effect=economic_effect,
            position_changed=position_changed,
            crop_count_delta=crop_count_delta,
            weed_count_delta=weed_count_delta,
            operational_effect=operational_effect,
            expected_effect=expected_effect,
            observed_effect=observed_effect,
            agent_effect=agent_effect,
            environment_effect=environment_effect,
            effect_match=effect_match,
            evidence=dict(outcome),
        )

    @staticmethod
    def _executed(
        outcome: Dict[str, Any],
    ) -> bool:

        action = outcome.get("executed_action")

        if not action:
            return False

        farmer = action.get("farmer", [])
        hands = action.get("hands", [])
        market = action.get("market", [])

        return bool(farmer or hands or market)

    @staticmethod
    def _farmer_actions(
        outcome: Dict[str, Any],
    ) -> list:

        action = outcome.get("executed_action") or {}

        if not isinstance(action, dict):
            return []

        farmer = action.get("farmer", [])

        if isinstance(farmer, list):
            return farmer

        if farmer is None:
            return []

        return [farmer]

    @staticmethod
    def _market_actions(
        outcome: Dict[str, Any],
    ) -> list:

        action = outcome.get("executed_action") or {}

        if not isinstance(action, dict):
            return []

        market = action.get("market", [])

        if isinstance(market, list):
            return market

        if market is None:
            return []

        return [market]

    @classmethod
    def _farmer_action_name(
        cls,
        outcome: Dict[str, Any],
    ) -> str | None:

        actions = cls._farmer_actions(outcome)

        if not actions:
            return None

        action = actions[0]

        if isinstance(action, (list, tuple)):
            if not action:
                return None

            return str(action[0]).upper()

        return str(action).upper()

    @classmethod
    def _market_action_name(
        cls,
        outcome: Dict[str, Any],
    ) -> str | None:

        actions = cls._market_actions(outcome)

        if not actions:
            return None

        action = actions[0]

        if isinstance(action, (list, tuple)):
            if not action:
                return None

            return str(action[0]).upper()

        return str(action).upper()

    @classmethod
    def _is_terminal_farmer_action(
        cls,
        outcome: Dict[str, Any],
    ) -> bool:

        action_name = cls._farmer_action_name(outcome)

        return action_name in {
            "BUY_SEED",
            "PLANT",
            "WATER",
            "HARVEST",
            "DIG",
        }

    @staticmethod
    def _execution_effect(
        executed: bool,
    ) -> str:

        if executed:
            return "An executable action was recorded."

        return "No executable action was recorded."

    @staticmethod
    def _economic_effect(
        money_delta: float,
    ) -> str:

        if money_delta > 0:
            return "Immediate cash increased."

        if money_delta < 0:
            return "Immediate cash decreased."

        return "No immediate cash change."

    @staticmethod
    def _operational_effect(
        position_changed: bool,
        crop_count_delta: int,
        weed_count_delta: int,
    ) -> str:

        effects = []

        if position_changed:
            effects.append("farmer position changed")

        if crop_count_delta > 0:
            effects.append("crop count increased")
        elif crop_count_delta < 0:
            effects.append("crop count decreased")

        if weed_count_delta > 0:
            effects.append("weed count increased")
        elif weed_count_delta < 0:
            effects.append("weed count decreased")

        if not effects:
            return "No tracked operational change."

        return "; ".join(effects) + "."

    @staticmethod
    def _expected_effect(
        decision_name: str,
    ) -> str:

        name = decision_name.upper()

        if name == "PASS":
            return "No tracked state change is expected."

        if (
            name.startswith("BUY_")
            and name.endswith("_SEED")
        ):
            crop = name[
                len("BUY_"):-len("_SEED")
            ]

            return (
                f"{crop.title()} seed inventory should "
                "increase and cash should decrease."
            )

        if name.startswith("PLANT_"):
            crop = name[len("PLANT_"):]

            return (
                f"{crop.title()} seed inventory should "
                f"decrease and a {crop.lower()} crop "
                "should appear."
            )

        if name == "WATER_CROP":
            return (
                "The current crop should transition "
                "from not watered today to watered today."
            )

        if name == "HARVEST_CROP":
            return (
                "The current crop should be removed "
                "or its yield reduced, and harvested "
                "product should appear in the farmer's "
                "immediate inventory before later transfer "
                "to the farm shed."
            )

        if name == "DIG_WEED":
            return (
                "A weed on the current tile should "
                "be removed."
            )

        if name == "CLEAR_WEED":
            return (
                "The farmer should move toward the "
                "selected weed target; weed removal "
                "may occur on a later decision cycle."
            )

        if name == "SELL_WHEAT":
            return (
                "Wheat should be removed from the farm "
                "shed and cash should increase."
            )

        return (
            "The decision should produce its "
            "corresponding environment transition."
        )

    def _observed_effect(
        self,
        outcome: Dict[str, Any],
        money_delta: float,
        position_changed: bool,
        crop_count_delta: int,
        weed_count_delta: int,
    ) -> str:

        effects = []

        if money_delta != 0:
            effects.append(
                f"money delta: {money_delta:.1f}"
            )

        if position_changed:
            effects.append("farmer position changed")

        if crop_count_delta != 0:
            effects.append(
                f"crop count delta: {crop_count_delta}"
            )

        if weed_count_delta != 0:
            effects.append(
                f"weed count delta: {weed_count_delta}"
            )

        seed_changes = self._seed_deltas(outcome)

        for crop, delta in sorted(seed_changes.items()):
            if delta != 0:
                effects.append(
                    f"{crop} seed delta: {delta:.1f}"
                )

        shed_changes = self._shed_deltas(outcome)

        if shed_changes:
            effects.append(
                f"shed delta: {shed_changes}"
            )

        watered_change = (
            self._current_crop_water_state(outcome)
        )

        if watered_change is not None:
            before, after = watered_change
            effects.append(
                "Current crop watered_today: "
                f"{before} → {after}"
            )

        target_crop_change = self._target_crop_change(
            outcome
        )

        if target_crop_change is not None:
            before_crop, after_crop = target_crop_change

            if before_crop is not None:
                before_name = before_crop.get(
                    "crop",
                    before_crop.get("type"),
                )
            else:
                before_name = None

            if after_crop is not None:
                after_name = after_crop.get(
                    "crop",
                    after_crop.get("type"),
                )
            else:
                after_name = None

            if before_name != after_name:
                effects.append(
                    "target crop changed: "
                    f"{before_name} → {after_name}"
                )

        target_weed_change = self._target_weed_change(
            outcome
        )

        if target_weed_change is not None:
            before_weed, after_weed = target_weed_change

            if before_weed and not after_weed:
                effects.append(
                    "target weed removed"
                )

        if not effects:
            return "No tracked state change."

        return "; ".join(effects) + "."

    def _agent_effect(
        self,
        decision_name: str,
        outcome: Dict[str, Any],
        money_delta: float,
        position_changed: bool,
        crop_count_delta: int,
        weed_count_delta: int,
    ) -> str:

        name = decision_name.upper()

        if name == "PASS":
            return "NONE"

        if (
            name.startswith("BUY_")
            and name.endswith("_SEED")
        ):
            crop = name[
                len("BUY_"):-len("_SEED")
            ]

            seed_delta = (
                self._seed_deltas(outcome).get(
                    crop,
                    0.0,
                )
            )

            if seed_delta > 0 and money_delta < 0:
                return (
                    "BUY_SEED increased the requested "
                    "seed inventory and reduced cash."
                )

            if seed_delta > 0:
                return (
                    "BUY_SEED increased seed inventory."
                )

            if money_delta < 0:
                return "BUY_SEED reduced cash."

            return "NONE"

        if name.startswith("PLANT_"):
            crop = name[len("PLANT_"):]

            seed_delta = (
                self._seed_deltas(outcome).get(
                    crop,
                    0.0,
                )
            )

            effects = []

            if seed_delta < 0:
                effects.append(
                    f"{crop} seed consumed"
                )

            if crop_count_delta > 0:
                effects.append(
                    f"{crop} crop created"
                )

            if position_changed:
                effects.append(
                    "farmer moved toward planting target"
                )

            if effects:
                return (
                    "PLANT action effect: "
                    + "; ".join(effects)
                    + "."
                )

            return "NONE"

        if name == "WATER_CROP":
            watered_change = (
                self._current_crop_water_state(outcome)
            )

            if watered_change == (False, True):
                return (
                    "WATER action effect: current crop "
                    "changed to watered_today=True."
                )

            return "NONE"

        if name == "HARVEST_CROP":
            shed_delta = self._shed_deltas(
                outcome
            )

            harvested_product = outcome.get(
                "harvested_product"
            )
            harvested_units_delta = outcome.get(
                "harvested_units_delta",
                0,
            )

            effects = []

            if crop_count_delta < 0:
                effects.append("crop removed")

            if self._target_crop_removed(outcome):
                effects.append(
                    "target crop removed"
                )

            if (
                harvested_product is not None
                and harvested_units_delta > 0
            ):
                effects.append(
                    "harvested product added to "
                    "farmer inventory: "
                    f"{harvested_product} +"
                    f"{harvested_units_delta}"
                )

            if shed_delta:
                effects.append(
                    "harvested product added to shed: "
                    f"{shed_delta}"
                )

            if position_changed:
                effects.append(
                    "farmer moved toward harvest target"
                )

            if effects:
                return (
                    "HARVEST action effect: "
                    + "; ".join(effects)
                    + "."
                )

            return "NONE"

        if name == "DIG_WEED":
            if (
                weed_count_delta < 0
                or self._target_weed_removed(outcome)
            ):
                return (
                    "DIG action effect: weed removed."
                )

            return "NONE"

        if name == "CLEAR_WEED":
            if position_changed:
                return (
                    "CLEAR_WEED action effect: farmer "
                    "moved toward selected weed target."
                )

            return "NONE"

        if name == "SELL_WHEAT":
            shed_delta = self._shed_deltas(outcome)

            wheat_delta = shed_delta.get("WHEAT", 0)

            if wheat_delta < 0 and money_delta > 0:
                return (
                    "SELL action effect: wheat removed "
                    "from shed and cash increased."
                )

            if wheat_delta < 0:
                return (
                    "SELL action effect: wheat removed "
                    "from shed."
                )

            if money_delta > 0:
                return (
                    "SELL action effect: cash increased."
                )

            return "NONE"

        if position_changed:
            return (
                "Movement changed the farmer position."
            )

        return "NONE"

    def _environment_effect(
        self,
        decision_name: str,
        outcome: Dict[str, Any],
    ) -> str:

        name = decision_name.upper()
        effects = []

        watered_change = (
            self._current_crop_water_state(outcome)
        )

        if (
            watered_change == (True, False)
            and self._day_boundary(outcome)
        ):
            effects.append(
                "DAY_RESET: watered_today reset "
                "from True to False."
            )

        if (
            self._day_boundary(outcome)
            and self._weed_lifecycle_change(outcome)
        ):
            effects.append(
                "CROP_LIFECYCLE: crop/weed lifecycle "
                "transition occurred."
            )

        if (
            self._day_boundary(outcome)
            and self._unrelated_crop_change(
                decision_name=name,
                outcome=outcome,
            )
        ):
            effects.append(
                "CROP_LIFECYCLE: unrelated crop state "
                "changed during environment progression."
            )

        if not effects:
            return "NONE"

        return " ".join(effects)

    @staticmethod
    def _day_boundary(
        outcome: Dict[str, Any],
    ) -> bool:

        before_day = outcome.get("day_before")
        after_day = outcome.get("day_after")

        if (
            before_day is not None
            and after_day is not None
        ):
            return after_day > before_day

        before_step = outcome.get("step_before")
        after_step = outcome.get("step_after")

        if (
            before_step is not None
            and after_step is not None
        ):
            return (
                after_step > before_step
                and after_step % 24 == 0
            )

        return False

    @staticmethod
    def _weed_lifecycle_change(
        outcome: Dict[str, Any],
    ) -> bool:

        before = outcome.get("crops_before", [])
        after = outcome.get("crops_after", [])

        before_keys = {
            (
                crop.get("crop"),
                crop.get("row"),
                crop.get("col"),
            )
            for crop in before
        }

        after_keys = {
            (
                crop.get("crop"),
                crop.get("row"),
                crop.get("col"),
            )
            for crop in after
        }

        return bool(before_keys - after_keys)

    def _unrelated_crop_change(
        self,
        decision_name: str,
        outcome: Dict[str, Any],
    ) -> bool:

        if decision_name in {
            "PLANT_WHEAT",
            "PLANT_CARROT",
            "PLANT_TOMATO",
            "PLANT_STRAWBERRY",
            "PLANT_MELON",
            "HARVEST_CROP",
        }:
            return False

        return (
            outcome.get("crop_count_before", 0)
            != outcome.get("crop_count_after", 0)
        )

    @staticmethod
    def _tile_coordinates(
        outcome: Dict[str, Any],
    ):
        position = outcome.get("position_before")

        if (
            not isinstance(position, (list, tuple))
            or len(position) != 2
        ):
            return None

        return position[0], position[1]

    @classmethod
    def _crop_at_position(
        cls,
        crops,
        position,
    ):
        if not isinstance(crops, list):
            return None

        if position is None:
            return None

        x, y = position

        for crop in crops:
            if not isinstance(crop, dict):
                continue

            if (
                crop.get("col") == x
                and crop.get("row") == y
            ):
                return crop

        return None

    @classmethod
    def _target_crop_change(
        cls,
        outcome: Dict[str, Any],
    ):
        position = cls._tile_coordinates(outcome)

        if position is None:
            return None

        before = cls._crop_at_position(
            outcome.get("crops_before", []),
            position,
        )

        after = cls._crop_at_position(
            outcome.get("crops_after", []),
            position,
        )

        if before is None and after is None:
            return None

        return before, after

    @classmethod
    def _target_crop_removed(
        cls,
        outcome: Dict[str, Any],
    ) -> bool:

        change = cls._target_crop_change(outcome)

        if change is None:
            return False

        before, after = change

        if before is None:
            return False

        if after is None:
            return True

        before_name = before.get(
            "crop",
            before.get("type"),
        )

        after_name = after.get(
            "crop",
            after.get("type"),
        )

        return (
            before_name is not None
            and before_name != after_name
        )

    @classmethod
    def _target_weed_change(
        cls,
        outcome: Dict[str, Any],
    ):
        position = cls._tile_coordinates(outcome)

        if position is None:
            return None

        before_crops = outcome.get(
            "crops_before",
            [],
        )

        after_crops = outcome.get(
            "crops_after",
            [],
        )

        x, y = position

        before_weed = cls._weed_at_position(
            before_crops,
            x,
            y,
        )

        after_weed = cls._weed_at_position(
            after_crops,
            x,
            y,
        )

        if before_weed is None and after_weed is None:
            return None

        return before_weed, after_weed

    @staticmethod
    def _weed_at_position(
        crops,
        x,
        y,
    ):
        if not isinstance(crops, list):
            return None

        for crop in crops:
            if not isinstance(crop, dict):
                continue

            if (
                crop.get("col") != x
                or crop.get("row") != y
            ):
                continue

            crop_name = str(
                crop.get(
                    "crop",
                    crop.get("type", ""),
                )
            ).upper()

            if crop_name == "WEED":
                return crop

            if crop.get("weed") is True:
                return crop

        return None

    @classmethod
    def _target_weed_removed(
        cls,
        outcome: Dict[str, Any],
    ) -> bool:

        change = cls._target_weed_change(outcome)

        if change is None:
            return False

        before, after = change

        return before is not None and after is None

    def _effect_match(
        self,
        decision_name: str,
        outcome: Dict[str, Any],
        executed: bool,
        money_delta: float,
        position_changed: bool,
        crop_count_delta: int,
        weed_count_delta: int,
    ) -> str:

        name = decision_name.upper()

        if not executed:
            return "NOT_MATCHED"

        if name == "PASS":
            environment_effect = (
                self._environment_effect(
                    decision_name=name,
                    outcome=outcome,
                )
            )

            if environment_effect != "NONE":
                return "MATCHED"

            if self._has_tracked_change(outcome):
                return "NOT_MATCHED"

            return "MATCHED"

        if (
            name.startswith("BUY_")
            and name.endswith("_SEED")
        ):
            crop = name[
                len("BUY_"):-len("_SEED")
            ]

            seed_delta = (
                self._seed_deltas(outcome).get(
                    crop,
                    0.0,
                )
            )

            seed_ok = seed_delta > 0
            money_ok = money_delta < 0

            if seed_ok and money_ok:
                return "MATCHED"

            if seed_ok or money_ok:
                return "PARTIAL"

            return "NOT_MATCHED"

        if name.startswith("PLANT_"):
            crop = name[len("PLANT_"):]

            seed_delta = (
                self._seed_deltas(outcome).get(
                    crop,
                    0.0,
                )
            )

            seed_ok = seed_delta < 0
            crop_ok = crop_count_delta > 0

            if seed_ok and crop_ok:
                return "MATCHED"

            if seed_ok or crop_ok:
                return "PARTIAL"

            if position_changed:
                return "PARTIAL"

            return "NOT_MATCHED"

        if name == "WATER_CROP":
            watered_change = (
                self._current_crop_water_state(outcome)
            )

            if watered_change == (False, True):
                return "MATCHED"

            return "NOT_MATCHED"

        if name == "HARVEST_CROP":
            shed_delta = self._shed_deltas(
                outcome
            )

            crop_removed = (
                crop_count_delta < 0
                or self._target_crop_removed(outcome)
            )

            target_yield_reduced = bool(
                outcome.get(
                    "target_yield_reduced",
                    False,
                )
            )

            harvested_units_delta = outcome.get(
                "harvested_units_delta",
                0,
            )

            harvested_product_created = (
                isinstance(
                    harvested_units_delta,
                    (int, float),
                )
                and harvested_units_delta > 0
            )

            harvest_state_changed = (
                crop_removed
                or target_yield_reduced
            )

            terminal_action = (
                self._farmer_action_name(outcome)
                == "HARVEST"
            )

            # Kaggriculture puts the harvested product into
            # private["inventories"][0] immediately. Transfer
            # to the shed happens later at end-of-day.
            if (
                terminal_action
                and harvest_state_changed
                and harvested_product_created
            ):
                return "MATCHED"

            if (
                harvest_state_changed
                and harvested_product_created
            ):
                return "MATCHED"

            # Retain shed evidence as a backward-compatible
            # fallback for older outcome records.
            shed_created = any(
                delta > 0
                for delta in shed_delta.values()
            )

            if harvest_state_changed and shed_created:
                return "MATCHED"

            if (
                harvest_state_changed
                or harvested_product_created
                or shed_created
            ):
                return "PARTIAL"

            if position_changed:
                return "PARTIAL"

            return "NOT_MATCHED"

        if name == "DIG_WEED":
            weed_removed = (
                weed_count_delta < 0
                or self._target_weed_removed(outcome)
            )

            terminal_action = (
                self._farmer_action_name(outcome)
                == "DIG"
            )

            if terminal_action and weed_removed:
                return "MATCHED"

            if weed_removed:
                return "MATCHED"

            if terminal_action:
                return "PARTIAL"

            return "NOT_MATCHED"

        if name == "CLEAR_WEED":
            if position_changed:
                return "PARTIAL"

            return "NOT_APPLICABLE"

        if name == "SELL_WHEAT":
            shed_delta = self._shed_deltas(outcome)

            wheat_delta = shed_delta.get("WHEAT", 0)

            terminal_action = (
                self._market_action_name(outcome)
                == "SELL"
            )

            shed_ok = wheat_delta < 0
            money_ok = money_delta > 0

            if terminal_action and shed_ok and money_ok:
                return "MATCHED"

            if shed_ok and money_ok:
                return "MATCHED"

            if shed_ok or money_ok:
                return "PARTIAL"

            if terminal_action:
                return "PARTIAL"

            return "NOT_MATCHED"

        if name == "MOVE":
            if position_changed:
                return "MATCHED"

            return "NOT_MATCHED"

        if position_changed:
            return "MATCHED"

        if crop_count_delta != 0:
            return "MATCHED"

        if weed_count_delta != 0:
            return "MATCHED"

        return "NOT_APPLICABLE"

    @staticmethod
    def _has_tracked_change(
        outcome: Dict[str, Any],
    ) -> bool:

        tracked_pairs = (
            ("money_before", "money_after"),
            ("position_before", "position_after"),
            ("crop_count_before", "crop_count_after"),
            ("weed_count_before", "weed_count_after"),
            ("seeds_before", "seeds_after"),
            ("shed_before", "shed_after"),
            ("crops_before", "crops_after"),
        )

        return any(
            outcome.get(before) != outcome.get(after)
            for before, after in tracked_pairs
        )

    @staticmethod
    def _seed_deltas(
        outcome: Dict[str, Any],
    ) -> Dict[str, float]:

        before = outcome.get("seeds_before", {})
        after = outcome.get("seeds_after", {})

        crop_names = set(before) | set(after)

        return {
            crop: float(
                after.get(crop, 0)
                - before.get(crop, 0)
            )
            for crop in crop_names
        }

    @staticmethod
    def _shed_deltas(
        outcome: Dict[str, Any],
    ) -> Dict[str, int]:

        before = outcome.get("shed_before", {})
        after = outcome.get("shed_after", {})

        item_names = set(before) | set(after)

        return {
            item: int(
                after.get(item, 0)
                - before.get(item, 0)
            )
            for item in item_names
            if (
                after.get(item, 0)
                - before.get(item, 0)
            ) != 0
        }

    @staticmethod
    def _current_crop_water_state(
        outcome: Dict[str, Any],
    ):
        position = outcome.get("position_before")

        if not position:
            return None

        crops_before = outcome.get(
            "crops_before",
            [],
        )

        crops_after = outcome.get(
            "crops_after",
            [],
        )

        x, y = position

        before_crop = next(
            (
                crop
                for crop in crops_before
                if crop.get("col") == x
                and crop.get("row") == y
            ),
            None,
        )

        after_crop = next(
            (
                crop
                for crop in crops_after
                if crop.get("col") == x
                and crop.get("row") == y
            ),
            None,
        )

        if (
            before_crop is None
            or after_crop is None
        ):
            return None

        before = before_crop.get(
            "watered_today"
        )
        after = after_crop.get(
            "watered_today"
        )

        if before == after:
            return None

        return before, after