from typing import Any, Dict, Optional

from nexus.agents.nexus.agent import NexusAgent
from nexus.core.memory.experience_builder import ExperienceBuilder
from nexus.core.memory.experience_store import ExperienceStore
from nexus.environments.kaggriculture.environment import (
    KaggricultureEnvironment,
)
from nexus.evaluation.decision_intelligence import (
    DecisionIntelligenceBuilder,
)
from nexus.evaluation.metrics import evaluate_episode
from nexus.evaluation.outcomes import DecisionOutcomeEvaluator


class NexusAgentRunner:
    """
    Execute a NEXUS agent inside an environment.

    The runner orchestrates the complete decision cycle:

        observe
            ↓
        decide
            ↓
        plan
            ↓
        act
            ↓
        observe outcome
            ↓
        evaluate outcome
            ↓
        explain decision
            ↓
        build experience
            ↓
        store experience

    Experience records are descriptive learning artifacts.
    They do not imply that NEXUS has learned a policy yet.

    The runner also records target-level execution evidence for
    persistent spatial intents such as planting and harvesting.
    """

    def __init__(
        self,
        episode_steps: int = 720,
        player_id: int = 0,
        debug: bool = True,
        seed: int | None = None,
    ):
        self.episode_steps = episode_steps
        self.player_id = player_id
        self.seed = seed

        self.environment = KaggricultureEnvironment(
            episode_steps=episode_steps,
            debug=debug,
            seed=seed,
        )

        self.experience_store = ExperienceStore()

        self.agent = NexusAgent(
            player_id=player_id,
            experience_store=self.experience_store,
        )

        self.outcome_evaluator = DecisionOutcomeEvaluator()

        self.decision_intelligence_builder = (
            DecisionIntelligenceBuilder()
        )

        self.experience_builder = ExperienceBuilder()

    def _opponent_action(self) -> Dict[str, Any]:
        """Return a deterministic inactive opponent."""

        return {
            "farmer": ["PASS"],
            "hands": [],
            "market": [],
        }

    @staticmethod
    def _farm_snapshot(
        observation: Dict[str, Any],
        player_id: int,
    ) -> Dict[str, Any]:
        """
        Extract observable farm state needed for outcome tracking.

        NEXUS tracks canonical farm resources through:
            - private["seeds"]
            - private["shed"]

        Generic private["inventories"] is intentionally not tracked
        because it is not part of the established NEXUS observation
        contract and can produce misleading inventory deltas.
        """

        farm = observation["farms"][player_id]
        private = observation["private"]

        crop_count = 0
        weed_count = 0
        crops = []

        for row_index, row in enumerate(farm["tiles"]):
            for col_index, tile in enumerate(row):
                if not isinstance(tile, dict):
                    continue

                kind = tile.get("kind")

                if kind == "PLANT":
                    crop_count += 1

                    crops.append(
                        {
                            "crop": tile.get(
                                "crop",
                                "UNKNOWN",
                            ),
                            "row": row_index,
                            "col": col_index,
                            "planted_day": tile.get(
                                "planted_day",
                                0,
                            ),
                            "watered_today": tile.get(
                                "watered_today",
                                False,
                            ),
                            "yield_units": tile.get(
                                "yield_units",
                                0,
                            ),
                            "fertilized_until_day": tile.get(
                                "fertilized_until_day",
                                -1,
                            ),
                        }
                    )

                elif kind == "WEED":
                    weed_count += 1

        return {
            "money": float(
                farm["money"]
            ),
            "farmer_position": list(
                farm["farmer"]
            ),
            "crop_count": crop_count,
            "weed_count": weed_count,
            "seeds": dict(
                private["seeds"]
            ),
            "shed": dict(
                private["shed"]
            ),
            "crops": crops,
        }

    @staticmethod
    def _crop_at_target(
        observation: Dict[str, Any],
        player_id: int,
        target: Optional[Dict[str, int]],
    ) -> Optional[Dict[str, Any]]:
        """
        Return the crop occupying a persistent execution target.

        ExecutionIntent coordinates use:
            x = column
            y = row

        Kaggriculture tile arrays use:
            tiles[row][column]

        A copied dictionary is returned so diagnostic evidence
        cannot mutate the environment observation.
        """

        if not target:
            return None

        if "x" not in target or "y" not in target:
            return None

        x = target["x"]
        y = target["y"]

        farm = observation["farms"][player_id]
        tiles = farm.get("tiles", [])

        if y < 0 or y >= len(tiles):
            return None

        row = tiles[y]

        if x < 0 or x >= len(row):
            return None

        tile = row[x]

        if not isinstance(tile, dict):
            return None

        if tile.get("kind") != "PLANT":
            return None

        return {
            "crop": tile.get(
                "crop",
                "UNKNOWN",
            ),
            "x": x,
            "y": y,
            "row": y,
            "col": x,
            "planted_day": tile.get(
                "planted_day",
                0,
            ),
            "watered_today": tile.get(
                "watered_today",
                False,
            ),
            "yield_units": tile.get(
                "yield_units",
                0,
            ),
            "fertilized_until_day": tile.get(
                "fertilized_until_day",
                -1,
            ),
        }

    def _current_execution_target(
        self,
    ) -> Optional[Dict[str, int]]:
        """
        Capture the target currently owned by the agent's persistent
        ExecutionIntent.

        The target is read immediately after agent.act() and before
        the environment executes the low-level action.

        This preserves the exact target associated with the action
        being sent to Kaggriculture.
        """

        execution_intent = getattr(
            self.agent,
            "execution_intent",
            None,
        )

        if execution_intent is None:
            return None

        target = getattr(
            execution_intent,
            "target",
            None,
        )

        if not isinstance(target, dict):
            return None

        if "x" not in target or "y" not in target:
            return None

        try:
            return {
                "x": int(target["x"]),
                "y": int(target["y"]),
            }
        except (
            TypeError,
            ValueError,
        ):
            return None

    @staticmethod
    def _target_evidence(
        before_observation: Dict[str, Any],
        after_observation: Dict[str, Any],
        player_id: int,
        target: Optional[Dict[str, int]],
    ) -> Dict[str, Any]:
        """
        Build exact before/after evidence for a persistent spatial target.

        This is especially important for HARVEST_CROP because aggregate
        crop counts cannot prove that the intended crop at the intended
        coordinate was harvested.
        """

        if target is None:
            return {
                "execution_target": None,
                "target_crop_before": None,
                "target_crop_after": None,
                "target_yield_before": None,
                "target_yield_after": None,
                "target_crop_removed": False,
                "target_yield_reduced": False,
            }

        before_crop = NexusAgentRunner._crop_at_target(
            observation=before_observation,
            player_id=player_id,
            target=target,
        )

        after_crop = NexusAgentRunner._crop_at_target(
            observation=after_observation,
            player_id=player_id,
            target=target,
        )

        before_yield = (
            before_crop.get("yield_units")
            if before_crop is not None
            else None
        )

        after_yield = (
            after_crop.get("yield_units")
            if after_crop is not None
            else None
        )

        target_crop_removed = (
            before_crop is not None
            and after_crop is None
        )

        target_yield_reduced = (
            before_yield is not None
            and after_yield is not None
            and after_yield < before_yield
        )

        return {
            "execution_target": dict(target),
            "target_crop_before": before_crop,
            "target_crop_after": after_crop,
            "target_yield_before": before_yield,
            "target_yield_after": after_yield,
            "target_crop_removed": target_crop_removed,
            "target_yield_reduced": target_yield_reduced,
        }

    @staticmethod
    def _harvest_inventory_evidence(
        before_observation: Dict[str, Any],
        after_observation: Dict[str, Any],
        player_id: int,
        target_evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Capture only harvest-specific immediate inventory evidence.

        Kaggriculture places harvested products into
        private["inventories"][0] immediately. The product is transferred
        to private["shed"] later at end-of-day.

        Generic inventory tracking remains intentionally excluded from
        _farm_snapshot().
        """

        before_crop = target_evidence.get(
            "target_crop_before"
        )

        if not isinstance(before_crop, dict):
            return {
                "harvested_product": None,
                "harvested_units_before": None,
                "harvested_units_after": None,
                "harvested_units_delta": 0,
            }

        product = before_crop.get("crop")

        if not product:
            return {
                "harvested_product": None,
                "harvested_units_before": None,
                "harvested_units_after": None,
                "harvested_units_delta": 0,
            }

        def _main_inventory(
            observation: Dict[str, Any],
        ) -> Dict[str, Any]:
            private = observation.get("private")

            if private is None:
                return {}

            inventories = private.get("inventories", [])

            if not inventories:
                return {}

            inventory = inventories[0]

            return (
                dict(inventory)
                if isinstance(inventory, dict)
                else {}
            )

        before_inventory = _main_inventory(
            before_observation
        )
        after_inventory = _main_inventory(
            after_observation
        )

        before_units = int(
            before_inventory.get(product, 0)
        )
        after_units = int(
            after_inventory.get(product, 0)
        )

        return {
            "harvested_product": product,
            "harvested_units_before": before_units,
            "harvested_units_after": after_units,
            "harvested_units_delta": (
                after_units - before_units
            ),
        }

    def _build_outcome(
        self,
        before_observation: Dict[str, Any],
        after_observation: Dict[str, Any],
        action: Dict[str, Any],
        execution_target: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        Build descriptive before/after transition evidence.

        Outcome evidence is based only on canonical observable
        NEXUS state:
            - money
            - farmer position
            - crop state
            - weed state
            - seeds
            - shed

        Generic inventory snapshots are deliberately excluded.

        For persistent spatial execution, exact target-level evidence
        is also recorded.
        """

        before = self._farm_snapshot(
            before_observation,
            self.player_id,
        )

        after = self._farm_snapshot(
            after_observation,
            self.player_id,
        )

        target_evidence = self._target_evidence(
            before_observation=before_observation,
            after_observation=after_observation,
            player_id=self.player_id,
            target=execution_target,
        )

        harvest_inventory_evidence = {
            "harvested_product": None,
            "harvested_units_before": None,
            "harvested_units_after": None,
            "harvested_units_delta": 0,
        }

        farmer_action = action.get("farmer", [])

        if (
            isinstance(farmer_action, list)
            and farmer_action
            and farmer_action[0] == "HARVEST"
        ):
            harvest_inventory_evidence = (
                self._harvest_inventory_evidence(
                    before_observation=before_observation,
                    after_observation=after_observation,
                    player_id=self.player_id,
                    target_evidence=target_evidence,
                )
            )

        return {
            "step_before": before_observation["step"],
            "step_after": after_observation["step"],
            "day_before": before_observation["day"],
            "day_after": after_observation["day"],
            "hour_before": before_observation["hour"],
            "hour_after": after_observation["hour"],

            "money_before": before["money"],
            "money_after": after["money"],
            "money_delta": (
                after["money"]
                - before["money"]
            ),

            "position_before": before[
                "farmer_position"
            ],
            "position_after": after[
                "farmer_position"
            ],

            "crop_count_before": before[
                "crop_count"
            ],
            "crop_count_after": after[
                "crop_count"
            ],

            "weed_count_before": before[
                "weed_count"
            ],
            "weed_count_after": after[
                "weed_count"
            ],

            "seeds_before": before[
                "seeds"
            ],
            "seeds_after": after[
                "seeds"
            ],

            "shed_before": before[
                "shed"
            ],
            "shed_after": after[
                "shed"
            ],

            "crops_before": before[
                "crops"
            ],
            "crops_after": after[
                "crops"
            ],

            "execution_target": target_evidence[
                "execution_target"
            ],

            "target_crop_before": target_evidence[
                "target_crop_before"
            ],

            "target_crop_after": target_evidence[
                "target_crop_after"
            ],

            "target_yield_before": target_evidence[
                "target_yield_before"
            ],

            "target_yield_after": target_evidence[
                "target_yield_after"
            ],

            "target_crop_removed": target_evidence[
                "target_crop_removed"
            ],

            "target_yield_reduced": target_evidence[
                "target_yield_reduced"
            ],

            "harvested_product": harvest_inventory_evidence[
                "harvested_product"
            ],
            "harvested_units_before": harvest_inventory_evidence[
                "harvested_units_before"
            ],
            "harvested_units_after": harvest_inventory_evidence[
                "harvested_units_after"
            ],
            "harvested_units_delta": harvest_inventory_evidence[
                "harvested_units_delta"
            ],

            "executed_action": action,
        }

    def _evaluate_outcome(
        self,
        decision_name: str,
        outcome: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate observable environmental effects."""

        evaluation = self.outcome_evaluator.evaluate(
            decision_name=decision_name,
            outcome=outcome,
        )

        return evaluation.as_dict()

    def _build_decision_intelligence(
        self,
        decision_record,
        outcome_evaluation,
    ) -> Dict[str, Any]:
        """Build the serialized decision-intelligence report."""

        report = self.decision_intelligence_builder.build(
            record=decision_record,
            outcome=outcome_evaluation,
        )

        return report.as_dict()

    def _build_experience(
        self,
        decision_record,
        intelligence,
    ):
        """Convert decision evidence into one learning experience."""

        return self.experience_builder.build(
            record=decision_record,
            intelligence=type(
                "_DecisionIntelligenceReportProxy",
                (),
                {
                    "step": intelligence["step"],
                    "day": intelligence["day"],
                    "hour": intelligence["hour"],
                    "decision_name": intelligence[
                        "decision_name"
                    ],
                    "decision_action": intelligence[
                        "decision_action"
                    ],
                    "executed_action": intelligence[
                        "executed_action"
                    ],
                    "decision_score": intelligence[
                        "decision_score"
                    ],
                    "decision_reason": intelligence[
                        "decision_reason"
                    ],
                    "confidence": intelligence[
                        "confidence"
                    ],
                    "score_margin": intelligence[
                        "score_margin"
                    ],
                    "runner_up": intelligence[
                        "runner_up"
                    ],
                    "runner_up_score": intelligence[
                        "runner_up_score"
                    ],
                    "alternatives": [
                        type(
                            "_DecisionAlternativeProxy",
                            (),
                            alternative,
                        )()
                        for alternative in intelligence[
                            "alternatives"
                        ]
                    ],
                    "expected_effect": intelligence[
                        "expected_effect"
                    ],
                    "observed_effect": intelligence[
                        "observed_effect"
                    ],
                    "agent_effect": intelligence[
                        "agent_effect"
                    ],
                    "environment_effect": intelligence[
                        "environment_effect"
                    ],
                    "effect_match": intelligence[
                        "effect_match"
                    ],
                    "evidence": intelligence[
                        "evidence"
                    ],
                    "alternative_count": len(
                        intelligence[
                            "alternatives"
                        ]
                    ),
                },
            )(),
        )

    def run(self) -> Dict[str, Any]:
        """Run one complete NEXUS agent episode."""

        self.environment.reset()

        decisions = []
        decision_intelligence = []

        self.experience_store.clear()

        for _ in range(
            self.episode_steps
        ):
            if self.environment.state[
                self.player_id
            ].status != "ACTIVE":
                break

            observation = self.environment.state[
                self.player_id
            ].observation

            action = self.agent.act(
                observation
            )

            execution_target = (
                self._current_execution_target()
            )

            decisions.append(
                {
                    "step": observation[
                        "step"
                    ],
                    "day": observation[
                        "day"
                    ],
                    "hour": observation[
                        "hour"
                    ],
                    "action": action,
                    "execution_target": (
                        dict(execution_target)
                        if execution_target is not None
                        else None
                    ),
                }
            )

            self.environment.step(
                [
                    action,
                    self._opponent_action(),
                ]
            )

            post_action_observation = (
                self.environment.state[
                    self.player_id
                ].observation
            )

            outcome = self._build_outcome(
                before_observation=observation,
                after_observation=(
                    post_action_observation
                ),
                action=action,
                execution_target=execution_target,
            )

            decision_record = (
                self.agent.memory.latest()
            )

            if decision_record is None:
                raise RuntimeError(
                    "NEXUS memory did not contain "
                    "the decision corresponding to "
                    "the executed action."
                )

            evaluation_object = (
                self.outcome_evaluator.evaluate(
                    decision_name=(
                        decision_record.decision_name
                    ),
                    outcome=outcome,
                )
            )

            evaluation = (
                evaluation_object.as_dict()
            )

            outcome["evaluation"] = evaluation

            intelligence = (
                self._build_decision_intelligence(
                    decision_record=decision_record,
                    outcome_evaluation=(
                        evaluation_object
                    ),
                )
            )

            outcome["decision_intelligence"] = (
                intelligence
            )

            decision_intelligence.append(
                intelligence
            )

            self.agent.memory.record_outcome(
                outcome
            )

            updated_record = (
                self.agent.memory.latest()
            )

            if updated_record is None:
                raise RuntimeError(
                    "NEXUS memory lost the decision "
                    "record after recording the outcome."
                )

            experience = self._build_experience(
                decision_record=updated_record,
                intelligence=intelligence,
            )

            self.experience_store.add(
                experience
            )

        final_observation = (
            self.environment.state[
                self.player_id
            ].observation
        )

        final_money = (
            final_observation[
                "farms"
            ][self.player_id]["money"]
        )

        shed = (
            final_observation[
                "private"
            ]["shed"]
        )

        evaluation = evaluate_episode(
            starting_money=3000.0,
            final_money=final_money,
            shed=shed,
        )

        experiences = [
            experience.as_dict()
            for experience in self.experience_store.all()
        ]

        return {
            "observation": final_observation,
            "decisions": decisions,
            "memory": self.agent.memory.all(),
            "decision_intelligence": decision_intelligence,
            "experiences": experiences,
            "experience_count": len(
                experiences
            ),
            "evaluation": evaluation,
        }


def _print_execution_diagnostics(
    result: Dict[str, Any],
) -> None:
    """
    Print important real-environment execution transitions.

    This is diagnostic output only. It does not modify decisions.
    """

    intelligence_records = result.get(
        "decision_intelligence",
        [],
    )

    memory_records = result.get(
        "memory",
        [],
    )

    print()
    print("=" * 80)
    print("NEXUS EXECUTION DIAGNOSTICS")
    print("=" * 80)

    tracked = 0

    for record in intelligence_records:
        decision_name = record.get(
            "decision_name",
            "UNKNOWN",
        )

        if decision_name not in {
            "PLANT_WHEAT",
            "PLANT_CARROT",
            "PLANT_TOMATO",
            "PLANT_STRAWBERRY",
            "PLANT_MELON",
            "HARVEST_CROP",
        }:
            continue

        tracked += 1

        step = record.get(
            "step",
            "N/A",
        )

        action = record.get(
            "decision_action",
            "N/A",
        )

        effect_match = record.get(
            "effect_match",
            "N/A",
        )

        expected = record.get(
            "expected_effect",
            "N/A",
        )

        observed = record.get(
            "observed_effect",
            "N/A",
        )

        print()
        print(
            f"[STEP {step}] {decision_name}"
        )
        print(
            f"  decision action : {action}"
        )
        print(
            f"  effect match    : {effect_match}"
        )
        print(
            f"  expected        : {expected}"
        )
        print(
            f"  observed        : {observed}"
        )

        matching_memory = None

        for memory_record in reversed(
            memory_records
        ):
            if memory_record.get(
                "step"
            ) == step:
                matching_memory = memory_record
                break

        if matching_memory is not None:
            outcome = matching_memory.get(
                "outcome",
                {},
            )

            if not isinstance(
                outcome,
                dict,
            ):
                outcome = {}

            target = outcome.get(
                "execution_target"
            )

            if target is not None:
                print(
                    f"  target          : {target}"
                )

                print(
                    "  target before   : "
                    f"{outcome.get('target_crop_before')}"
                )

                print(
                    "  target after    : "
                    f"{outcome.get('target_crop_after')}"
                )

                print(
                    "  yield before    : "
                    f"{outcome.get('target_yield_before')}"
                )

                print(
                    "  yield after     : "
                    f"{outcome.get('target_yield_after')}"
                )

                print(
                    "  crop removed    : "
                    f"{outcome.get('target_crop_removed')}"
                )

                print(
                    "  yield reduced   : "
                    f"{outcome.get('target_yield_reduced')}"
                )

    if tracked == 0:
        print()
        print(
            "No planting or harvest decisions "
            "were recorded during this episode."
        )

    print()
    print("=" * 80)


def _print_final_state(
    result: Dict[str, Any],
) -> None:
    """Print final observable farm state."""

    observation = result.get(
        "observation",
        {},
    )

    farms = observation.get(
        "farms",
        [],
    )

    private = observation.get(
        "private",
        {},
    )

    if not farms:
        return

    farm = farms[0]

    print()
    print("=" * 80)
    print("NEXUS FINAL FARM STATE")
    print("=" * 80)

    print(
        f"Money:          {farm.get('money', 'N/A')}"
    )

    print(
        f"Farmer position: "
        f"{farm.get('farmer', 'N/A')}"
    )

    seeds = private.get(
        "seeds",
        {},
    )

    shed = private.get(
        "shed",
        {},
    )

    print(
        f"Seeds:          {seeds}"
    )

    print(
        f"Shed:           {shed}"
    )

    print("=" * 80)


def main() -> None:
    """
    Command-line entry point for a real NEXUS episode.
    """

    print("=" * 80)
    print(
        "NEXUS — Intelligent Farm Decision & "
        "Autonomous Management System"
    )
    print("=" * 80)
    print()
    print(
        "Starting NEXUS agent runner..."
    )
    print(
        "Episode steps: 720"
    )
    print(
        "Player ID:     0"
    )
    print(
        "Debug:         True"
    )
    print(
        "Seed:          101"
    )
    print()

    runner = NexusAgentRunner(
        episode_steps=720,
        player_id=0,
        debug=True,
        seed=101,
    )

    result = runner.run()

    evaluation = result.get(
        "evaluation",
        {},
    )

    _print_execution_diagnostics(
        result
    )

    _print_final_state(
        result
    )

    print()
    print("=" * 80)
    print("NEXUS RUN COMPLETE")
    print("=" * 80)

    print(
        f"Starting money: "
        f"{evaluation.get('starting_money', 3000.0)}"
    )

    print(
        f"Final money:    "
        f"{evaluation.get('final_money', 'N/A')}"
    )

    print(
        f"Net profit:     "
        f"{evaluation.get('net_profit', 'N/A')}"
    )

    print(
        f"ROI:            "
        f"{evaluation.get('roi_percent', 'N/A'):.2f}%"
    )

    print(
        f"Experiences:    "
        f"{result.get('experience_count', 0)}"
    )

    print(
        f"Decisions:      "
        f"{len(result.get('decisions', []))}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()