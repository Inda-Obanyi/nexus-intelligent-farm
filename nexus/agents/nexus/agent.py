from typing import Any, Dict, Optional

from nexus.core.decision.context import (
    DecisionContext,
    DecisionConstraints,
    DecisionGoal,
)
from nexus.core.decision.engine import DecisionEngine
from nexus.core.memory.decision import DecisionRecord
from nexus.core.memory.experience_retriever import ExperienceRetriever
from nexus.core.memory.experience_store import ExperienceStore
from nexus.core.memory.learning_signal import LearningSignalBuilder
from nexus.core.memory.evidence_interpreter import EvidenceInterpreter
from nexus.core.memory.store import DecisionMemory
from nexus.core.planning.execution_intent import ExecutionIntent
from nexus.core.planning.planner import FarmPlanner, PlanningTarget
from nexus.core.planning.target_selector import SpatialTargetSelector
from nexus.core.state import FarmState, NexusStateBuilder
from nexus.intelligence.risk.builder import RiskSignalBuilder


class NexusAgent:
    """
    NEXUS intelligent farm agent.

    Decision pipeline:

        Observation
            ↓
        State Building
            ↓
        Risk Perception
            ↓
        Decision Engine
            ↓
        Decision Trace
            ↓
        Strategic Execution Intent
            ↓
        Spatial Target Selection
            ↓
        Planning / Navigation
            ↓
        Environment Action
            ↓
        Memory / Audit

    Architectural responsibilities:

    DecisionEngine
        Determines WHAT should happen.

    SpatialTargetSelector
        Determines WHERE the selected objective should happen.

    FarmPlanner
        Determines HOW to reach the selected location.

    ExecutionIntent
        Maintains a strategic spatial objective across
        multiple environment timesteps.

    NexusAgent
        Coordinates the complete:

        observe → understand → decide → target →
        plan → execute → observe → remember

        cycle.
    """

    def __init__(
        self,
        player_id: int = 0,
        minimum_cash: float = 1000.0,
        experience_store: Optional[ExperienceStore] = None,
    ):
        self.player_id = player_id
        self.minimum_cash = minimum_cash

        self.state_builder = NexusStateBuilder(
            player_id=player_id
        )

        self.decision_engine = DecisionEngine()

        self.risk_signal_builder = RiskSignalBuilder()

        self.target_selector = SpatialTargetSelector()

        self.planner = FarmPlanner()

        self.memory = DecisionMemory()

        # Canonical historical experience store. The runner injects
        # its shared store; standalone agent tests get an empty store.
        self.experience_store = (
            experience_store
            if experience_store is not None
            else ExperienceStore()
        )
        self.experience_retriever = ExperienceRetriever(
            self.experience_store
        )
        self.learning_signal_builder = LearningSignalBuilder()
        self.evidence_interpreter = EvidenceInterpreter()

        self.planning_target: Optional[PlanningTarget] = None

        self.selected_spatial_target = None

        self.execution_intent: Optional[ExecutionIntent] = None

    # =========================================================
    # MAIN AGENT LOOP
    # =========================================================

    def act(
        self,
        observation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute one NEXUS decision/execution cycle.

        A spatial strategic decision creates an ExecutionIntent.
        The intent persists while the farmer navigates toward
        its target.

        Non-spatial decisions execute immediately.
        """

        state = self.state_builder.build(observation)

        # -----------------------------------------------------
        # Continue an existing strategic execution intent.
        # -----------------------------------------------------

        if (
            self.execution_intent is not None
            and self.execution_intent.is_active()
        ):
            lifecycle = self._evaluate_execution_intent(
                state
            )

            # -------------------------------------------------
            # Objective completed or failed.
            # -------------------------------------------------

            if lifecycle in {
                "COMPLETED",
                "FAILED",
            }:
                self.planning_target = None
                self.selected_spatial_target = None
                self.execution_intent = None

            # -------------------------------------------------
            # Continue existing strategic objective.
            # -------------------------------------------------

            else:
                (
                    farmer_action,
                    metadata,
                ) = self._continue_execution_intent(
                    state
                )

                action = {
                    "farmer": farmer_action,
                    "hands": [],
                    "market": [],
                }

                self.memory.add(
                    DecisionRecord(
                        step=state.raw_observation["step"],
                        day=state.day,
                        hour=state.hour,
                        observation_summary={
                            "money": state.resources.money,
                            "farmer_position": list(
                                state.farmer_position
                            ),
                            "crop_count": len(state.crops),
                            "weed_count": len(state.weeds),
                            "wheat_seeds": (
                                state.resources.seeds.get(
                                    "WHEAT",
                                    0,
                                )
                            ),
                            "wheat_inventory": (
                                state.resources.shed.get(
                                    "WHEAT",
                                    0,
                                )
                            ),
                        },
                        decision_name=(
                            self.execution_intent.decision_name
                        ),
                        decision_action=list(
                            self.execution_intent.action
                        ),
                        decision_score=0.0,
                        decision_reason=(
                            "Continuing an existing "
                            "strategic execution intent."
                        ),
                        decision_trace=None,
                        executed_action=action,
                        metadata=metadata,
                    )
                )

                return action

        # -----------------------------------------------------
        # No active intent → build decision context.
        # -----------------------------------------------------

        crop_risks = {
            crop.crop: self.risk_signal_builder.build(
                crop,
                state,
            )
            for crop in state.crops
        }

        context = DecisionContext(
            state=state,
            goals=[
                DecisionGoal(
                    name="maximize_profit",
                    priority=1.0,
                )
            ],
            constraints=DecisionConstraints(
                minimum_cash=self.minimum_cash
            ),
            crop_risks=crop_risks,
        )

        # -----------------------------------------------------
        # Strategic decision.
        # -----------------------------------------------------

        decision = self.decision_engine.decide(context)

        decision_trace = getattr(
            self.decision_engine,
            "last_trace",
            None,
        )

        # -----------------------------------------------------
        # Historical experience retrieval.
        #
        # This is intentionally POST-DECISION and descriptive.
        # The retrieved evidence must never modify the selected
        # decision, score, or action at this milestone.
        # -----------------------------------------------------

        retrieval_metadata = {
            "planning_target": self._retrieval_target(
                decision.action,
                state,
            ),
        }

        retrieved_experiences = self.experience_retriever.retrieve(
            observation_summary={
                "money": state.resources.money,
                "farmer_position": list(
                    state.farmer_position
                ),
                "crop_count": len(state.crops),
                "weed_count": len(state.weeds),
                "wheat_seeds": state.resources.seeds.get(
                    "WHEAT",
                    0,
                ),
                "wheat_inventory": state.resources.shed.get(
                    "WHEAT",
                    0,
                ),
            },
            decision_name=decision.name,
            decision_action=list(decision.action),
            metadata=retrieval_metadata,
        )

        learning_signal = self.learning_signal_builder.build(
            retrieved_experiences
        )

        # -----------------------------------------------------
        # Historical evidence interpretation.
        #
        # This runs AFTER the current decision has already been
        # selected. It is descriptive only and cannot modify
        # the decision score, selected action, or execution.
        # -----------------------------------------------------

        evidence_interpretation = (
            self.evidence_interpreter.interpret(
                retrieved_experiences
            )
        )

        retrieval_summary = (
            self.experience_retriever.summarize(
                retrieved_experiences
            )
        )

        context.learning_signal = learning_signal

        self.selected_spatial_target = None

        # -----------------------------------------------------
        # Spatial strategic decision.
        # -----------------------------------------------------

        if self._is_spatial_intent(decision.name):
            (
                farmer_action,
                metadata,
            ) = self._start_execution_intent(
                decision,
                state,
            )

            market_action = []

        # -----------------------------------------------------
        # Normal non-spatial decision.
        # -----------------------------------------------------

        else:
            market_action = self._market_action(
                decision.action
            )

            farmer_action = self._farmer_action(
                decision.action,
                state,
            )

            metadata = self._planning_metadata(
                decision.action,
                state,
                farmer_action,
            )

        action = {
            "farmer": farmer_action,
            "hands": [],
            "market": market_action,
        }

        metadata["learning_signal"] = (
            learning_signal.as_dict()
        )
        metadata["evidence_interpretation"] = (
            evidence_interpretation.as_dict()
        )
        metadata["retrieval_summary"] = (
            retrieval_summary
        )
        metadata["retrieved_experiences"] = [
            item.as_dict()
            for item in retrieved_experiences
        ]

        # -----------------------------------------------------
        # Store decision record.
        # -----------------------------------------------------

        self.memory.add(
            DecisionRecord(
                step=state.raw_observation["step"],
                day=state.day,
                hour=state.hour,
                observation_summary={
                    "money": state.resources.money,
                    "farmer_position": list(
                        state.farmer_position
                    ),
                    "crop_count": len(state.crops),
                    "weed_count": len(state.weeds),
                    "wheat_seeds": (
                        state.resources.seeds.get(
                            "WHEAT",
                            0,
                        )
                    ),
                    "wheat_inventory": (
                        state.resources.shed.get(
                            "WHEAT",
                            0,
                        )
                    ),
                },
                decision_name=decision.name,
                decision_action=list(
                    decision.action
                ),
                decision_score=decision.score,
                decision_reason=decision.reason,
                decision_trace=decision_trace,
                executed_action=action,
                metadata=metadata,
            )
        )

        return action

    # =========================================================
    # SPATIAL INTENT
    # =========================================================

    @staticmethod
    def _is_spatial_intent(
        decision_name: str,
    ) -> bool:
        """
        Determine whether a strategic decision requires
        persistent spatial execution.
        """

        if not decision_name:
            return False

        spatial_intents = {
            "CLEAR_WEED",
            "PLANT_WHEAT",
            "PLANT_CARROT",
            "PLANT_TOMATO",
            "PLANT_STRAWBERRY",
            "PLANT_MELON",
            "HARVEST_CROP",
        }

        return decision_name in spatial_intents

    # =========================================================
    # START EXECUTION INTENT
    # =========================================================

    def _start_execution_intent(
        self,
        decision,
        state: FarmState,
    ):
        target = None

        # -----------------------------------------------------
        # Planting target.
        # -----------------------------------------------------

        if decision.name.startswith("PLANT_"):
            spatial_target = (
                self.target_selector.nearest_plantable_tile(
                    state
                )
            )

            if spatial_target is None:
                self.execution_intent = None
                self.planning_target = None
                self.selected_spatial_target = None

                return (
                    ["PASS"],
                    {
                        "execution_intent": None,
                        "planning_target": None,
                        "planned_move": "PASS",
                        "spatial_target": None,
                        "spatial_target_reason": (
                            "No plantable tile available."
                        ),
                    },
                )

            target = {
                "x": int(spatial_target.x),
                "y": int(spatial_target.y),
            }

            self.selected_spatial_target = spatial_target

            self.planning_target = PlanningTarget(
                x=target["x"],
                y=target["y"],
            )

        # -----------------------------------------------------
        # Harvest target.
        # -----------------------------------------------------

        elif decision.name == "HARVEST_CROP":
            selected_target = (
                self.target_selector.nearest_harvestable_crop(
                    state
                )
            )

            if selected_target is None:
                self.execution_intent = None
                self.planning_target = None
                self.selected_spatial_target = None

                return (
                    ["PASS"],
                    {
                        "execution_intent": None,
                        "planning_target": None,
                        "planned_move": "PASS",
                        "spatial_target": None,
                        "spatial_target_reason": (
                            "No harvestable crop available."
                        ),
                    },
                )

            target = {
                "x": int(selected_target.x),
                "y": int(selected_target.y),
            }

            self.selected_spatial_target = selected_target

            self.planning_target = PlanningTarget(
                x=selected_target.x,
                y=selected_target.y,
            )

        # -----------------------------------------------------
        # Weed-clearing target.
        # -----------------------------------------------------

        elif decision.name == "CLEAR_WEED":
            if len(decision.action) < 3:
                self.execution_intent = None
                self.planning_target = None
                self.selected_spatial_target = None

                return (
                    ["PASS"],
                    {
                        "execution_intent": None,
                        "planning_target": None,
                        "planned_move": "PASS",
                        "spatial_target": None,
                        "spatial_target_reason": (
                            "CLEAR_WEED decision has no target."
                        ),
                    },
                )

            target = {
                "x": int(decision.action[1]),
                "y": int(decision.action[2]),
            }

            self.planning_target = PlanningTarget(
                x=target["x"],
                y=target["y"],
            )

        # -----------------------------------------------------
        # Create persistent execution intent.
        # -----------------------------------------------------

        self.execution_intent = ExecutionIntent(
            decision_name=decision.name,
            action=list(decision.action),
            target=target,
            initial_crop_count=len(state.crops),
            initial_weed_count=len(state.weeds),
            initial_seed_count=(
                self._intent_seed_count(
                    decision,
                    state,
                )
            ),
        )

        # -----------------------------------------------------
        # Execute first low-level step.
        # -----------------------------------------------------

        farmer_action = self._execute_persistent_intent(
            state
        )

        metadata = self._execution_metadata()

        return farmer_action, metadata

    # =========================================================
    # EXECUTION INTENT LIFECYCLE
    # =========================================================

    def _evaluate_execution_intent(
        self,
        state: FarmState,
    ) -> str:
        """
        Evaluate the current strategic execution intent.

        Returns:

            COMPLETED
                Strategic objective achieved.

            FAILED
                Objective can no longer be achieved.

            IN_PROGRESS
                Objective still requires execution.
        """

        intent = self.execution_intent

        if intent is None:
            return "IN_PROGRESS"

        decision_name = intent.decision_name

        # =====================================================
        # PLANTING INTENT
        # =====================================================

        if decision_name.startswith("PLANT_"):
            crop = decision_name.replace(
                "PLANT_",
                "",
                1,
            )

            # -------------------------------------------------
            # Strong completion signal:
            # intended crop exists at persistent target.
            # -------------------------------------------------

            if intent.target is not None:
                target_crop = self._crop_at_target(
                    state,
                    intent.target,
                )

                if target_crop == crop:
                    intent.complete()
                    return "COMPLETED"

            # -------------------------------------------------
            # Secondary completion signal:
            # crop count increased and intended crop exists.
            # -------------------------------------------------

            if (
                intent.initial_crop_count is not None
                and len(state.crops)
                > intent.initial_crop_count
            ):
                if any(
                    current_crop.crop == crop
                    for current_crop in state.crops
                ):
                    intent.complete()
                    return "COMPLETED"

            # -------------------------------------------------
            # Failure: target disappeared.
            # -------------------------------------------------

            if intent.target is not None:
                target_tile = state.tile_at(
                    x=int(intent.target["x"]),
                    y=int(intent.target["y"]),
                )

                if target_tile is None:
                    intent.fail()
                    return "FAILED"

                if target_tile.locked:
                    intent.fail()
                    return "FAILED"

            # -------------------------------------------------
            # Failure: no seed remains.
            # -------------------------------------------------

            if (
                intent.initial_seed_count is not None
                and state.resources.seeds.get(
                    crop,
                    0,
                ) <= 0
            ):
                intent.fail()
                return "FAILED"

            return "IN_PROGRESS"

        # =====================================================
        # HARVEST INTENT
        # =====================================================

        if decision_name == "HARVEST_CROP":
            if intent.target is None:
                intent.fail()
                return "FAILED"

            target_x = int(
                intent.target["x"]
            )

            target_y = int(
                intent.target["y"]
            )

            # -------------------------------------------------
            # Crop disappeared after harvest.
            # -------------------------------------------------

            target_crop = self._crop_at_target(
                state,
                intent.target,
            )

            if target_crop is None:
                intent.complete()
                return "COMPLETED"

            # -------------------------------------------------
            # Crop still exists but has no harvestable yield.
            #
            # This is especially important for ongoing crops,
            # which remain on the farm after harvest.
            # -------------------------------------------------

            for crop in state.crops:
                if (
                    crop.col == target_x
                    and crop.row == target_y
                ):
                    if crop.yield_units <= 0:
                        intent.complete()
                        return "COMPLETED"

                    return "IN_PROGRESS"

            # -------------------------------------------------
            # Target crop no longer exists.
            # -------------------------------------------------

            intent.complete()
            return "COMPLETED"

        # =====================================================
        # CLEAR WEED INTENT
        # =====================================================

        if decision_name == "CLEAR_WEED":
            if intent.target is None:
                intent.fail()
                return "FAILED"

            target_x = int(
                intent.target["x"]
            )

            target_y = int(
                intent.target["y"]
            )

            target_weed = self._weed_at_target(
                state,
                target_x,
                target_y,
            )

            if target_weed is None:
                intent.complete()
                return "COMPLETED"

            target_tile = state.tile_at(
                x=target_x,
                y=target_y,
            )

            if target_tile is None:
                intent.fail()
                return "FAILED"

            if target_tile.locked:
                intent.fail()
                return "FAILED"

            return "IN_PROGRESS"

        return "IN_PROGRESS"

    # =========================================================
    # STATE TARGET HELPERS
    # =========================================================

    def _crop_at_target(
        self,
        state: FarmState,
        target: Dict[str, int],
    ) -> Optional[str]:
        """
        Return the crop occupying the supplied target coordinate.
        """

        target_x = int(target["x"])
        target_y = int(target["y"])

        for crop in state.crops:
            if (
                crop.col == target_x
                and crop.row == target_y
            ):
                return crop.crop

        return None

    def _weed_at_target(
        self,
        state: FarmState,
        x: int,
        y: int,
    ):
        """
        Return the weed occupying [x, y], if one exists.
        """

        for weed in state.weeds:
            if (
                weed.col == x
                and weed.row == y
            ):
                return weed

        return None

    # =========================================================
    # CONTINUE EXECUTION
    # =========================================================

    def _continue_execution_intent(
        self,
        state: FarmState,
    ):
        """
        Continue the currently active strategic execution intent.
        """

        self._restore_intent_target()

        farmer_action = self._execute_persistent_intent(
            state
        )

        metadata = self._execution_metadata()

        return farmer_action, metadata

    def _restore_intent_target(self) -> None:
        """
        Restore the persistent spatial target stored on the
        active execution intent.
        """

        if self.execution_intent is None:
            return

        target = self.execution_intent.target

        if target is None:
            self.planning_target = None
            self.selected_spatial_target = None
            return

        self.planning_target = PlanningTarget(
            x=int(target["x"]),
            y=int(target["y"]),
        )

    # =========================================================
    # PERSISTENT EXECUTION
    # =========================================================

    def _execute_persistent_intent(
        self,
        state: FarmState,
    ):
        """
        Execute exactly one low-level environment action.
        """

        if self.execution_intent is None:
            return ["PASS"]

        decision_name = (
            self.execution_intent.decision_name
        )

        decision_action = list(
            self.execution_intent.action
        )

        if decision_name.startswith("PLANT_"):
            farmer_action = self._plant_action(
                decision_action,
                state,
            )

        elif decision_name == "CLEAR_WEED":
            farmer_action = self._clear_weed_action(
                decision_action,
                state,
            )

        elif decision_name == "HARVEST_CROP":
            farmer_action = self._harvest_action(
                decision_action,
                state,
            )

        else:
            farmer_action = ["PASS"]

        self.execution_intent.record_step(
            farmer_action
        )

        return farmer_action

    def _intent_seed_count(
        self,
        decision,
        state: FarmState,
    ) -> Optional[int]:
        """
        Capture relevant seed inventory when a planting
        intent starts.
        """

        if not decision.name.startswith("PLANT_"):
            return None

        crop = decision.name.replace(
            "PLANT_",
            "",
            1,
        )

        return state.resources.seeds.get(
            crop,
            0,
        )

    # =========================================================
    # EXECUTION METADATA
    # =========================================================

    def _execution_metadata(self) -> Dict[str, Any]:
        """
        Build metadata describing the persistent execution
        intent and latest low-level action.
        """

        metadata: Dict[str, Any] = {
            "execution_intent": None,
            "planning_target": None,
            "planned_move": None,
            "spatial_target": None,
            "spatial_target_reason": None,
        }

        if self.execution_intent is None:
            return metadata

        intent = self.execution_intent

        metadata["execution_intent"] = intent.as_dict()

        if intent.target is not None:
            metadata["planning_target"] = dict(
                intent.target
            )

            metadata["spatial_target"] = dict(
                intent.target
            )

        if self.selected_spatial_target is not None:
            metadata["spatial_target_reason"] = (
                self.selected_spatial_target.reason
            )

        if intent.steps:
            latest_step = intent.steps[-1]

            if latest_step:
                metadata["planned_move"] = (
                    latest_step[0]
                )

        return metadata

    # =========================================================
    # FARMER ACTION ROUTING
    # =========================================================

    def _farmer_action(
        self,
        decision_action,
        state: FarmState,
    ):
        """
        Convert strategic intent into an environment-compatible
        farmer action.
        """

        if not decision_action:
            return ["PASS"]

        action_name = decision_action[0]

        if action_name == "CLEAR_WEED":
            return self._clear_weed_action(
                decision_action,
                state,
            )

        if action_name == "HARVEST":
            return ["HARVEST"]

        plant_intent_names = {
            "PLANT_WHEAT",
            "PLANT_CARROT",
            "PLANT_TOMATO",
            "PLANT_STRAWBERRY",
            "PLANT_MELON",
        }

        is_plant_intent = (
            action_name in plant_intent_names
            or (
                action_name == "PLANT"
                and len(decision_action) >= 2
                and decision_action[1] in {
                    "WHEAT",
                    "CARROT",
                    "TOMATO",
                    "STRAWBERRY",
                    "MELON",
                }
            )
        )

        if is_plant_intent:
            return self._plant_action(
                decision_action,
                state,
            )

        if action_name == "DIG_WEED":
            self.planning_target = None
            self.selected_spatial_target = None
            return ["DIG"]

        if action_name == "PASS":
            self.planning_target = None
            self.selected_spatial_target = None
            return ["PASS"]

        self.planning_target = None
        self.selected_spatial_target = None

        return list(decision_action)

    # =========================================================
    # PLANT COMMAND
    # =========================================================

    def _plant_command(
        self,
        decision_action,
    ):
        """
        Preserve the complete crop-specific planting command.

        Canonical environment commands:

            ["PLANT", "WHEAT"]
            ["PLANT", "CARROT"]
            ["PLANT", "TOMATO"]
            ["PLANT", "STRAWBERRY"]
            ["PLANT", "MELON"]

        The crop argument MUST NOT be discarded.
        """

        # -----------------------------------------------------
        # Persistent execution intent has highest priority.
        # -----------------------------------------------------

        if self.execution_intent is not None:
            intent_action = list(
                self.execution_intent.action
            )

            if (
                len(intent_action) >= 2
                and intent_action[0] == "PLANT"
            ):
                return intent_action

            # -------------------------------------------------
            # Compatibility with strategic decision names.
            #
            # Example:
            #     decision_name = PLANT_WHEAT
            #     action = ["PLANT_WHEAT"]
            #
            # Convert this to the canonical environment command.
            # -------------------------------------------------

            if len(intent_action) == 1:
                strategic_name = intent_action[0]

                if strategic_name.startswith("PLANT_"):
                    crop = strategic_name.replace(
                        "PLANT_",
                        "",
                        1,
                    )

                    return [
                        "PLANT",
                        crop,
                    ]

        # -----------------------------------------------------
        # Direct decision action.
        # -----------------------------------------------------

        action = list(decision_action)

        if (
            len(action) >= 2
            and action[0] == "PLANT"
        ):
            return action

        # -----------------------------------------------------
        # Legacy strategic action.
        # -----------------------------------------------------

        if len(action) == 1:
            if action[0].startswith("PLANT_"):
                crop = action[0].replace(
                    "PLANT_",
                    "",
                    1,
                )

                return [
                    "PLANT",
                    crop,
                ]

        # -----------------------------------------------------
        # Backward-compatible generic planting command.
        # -----------------------------------------------------

        return ["PLANT"]

    # =========================================================
    # PLANTING
    # =========================================================

    def _plant_action(
        self,
        decision_action,
        state: FarmState,
    ):
        """
        Execute one planting-related low-level action.

        Persistent ExecutionIntent target takes priority.

        IMPORTANT:

        If the strategic command is:

            ["PLANT", "WHEAT"]

        the environment action remains:

            ["PLANT", "WHEAT"]

        and is never reduced to:

            ["PLANT"]
        """

        # -----------------------------------------------------
        # Persistent execution target.
        # -----------------------------------------------------

        if (
            self.execution_intent is not None
            and self.execution_intent.target is not None
        ):
            target_data = self.execution_intent.target

            self.planning_target = PlanningTarget(
                x=int(target_data["x"]),
                y=int(target_data["y"]),
            )

        # -----------------------------------------------------
        # Direct / legacy spatial planning.
        # -----------------------------------------------------

        else:
            self.planning_target = None

            target = (
                self.target_selector.nearest_plantable_tile(
                    state
                )
            )

            if target is None:
                self.selected_spatial_target = None
                return ["PASS"]

            self.selected_spatial_target = target

            self.planning_target = PlanningTarget(
                x=target.x,
                y=target.y,
            )

        # -----------------------------------------------------
        # Safety check.
        # -----------------------------------------------------

        if self.planning_target is None:
            return ["PASS"]

        current_position = list(
            state.farmer_position
        )

        target_position = list(
            self.planning_target.position
        )

        # -----------------------------------------------------
        # Farmer reached planting target.
        #
        # Preserve crop-specific command.
        # -----------------------------------------------------

        if current_position == target_position:
            return self._plant_command(
                decision_action
            )

        # -----------------------------------------------------
        # Navigate toward target.
        # -----------------------------------------------------

        next_move = self.planner.next_move(
            current_position=current_position,
            target=self.planning_target,
        )

        # -----------------------------------------------------
        # Planner cannot produce another movement step.
        #
        # Treat this as ready-to-execute rather than dropping
        # the crop argument.
        # -----------------------------------------------------

        if next_move is None:
            return self._plant_command(
                decision_action
            )

        return [next_move]

    # =========================================================
    # HARVESTING
    # =========================================================

    def _harvest_action(
        self,
        decision_action,
        state: FarmState,
    ):
        """
        Execute one harvest-related low-level action.

        The strategic HARVEST_CROP intent selects a specific
        harvestable crop and persists until the farmer reaches
        that crop.

        The environment receives:

            ["HARVEST"]

        only when the farmer reaches the selected crop.
        """

        if self.execution_intent is None:
            return ["PASS"]

        target = self.execution_intent.target

        if target is None:
            return ["PASS"]

        target_x = int(
            target["x"]
        )

        target_y = int(
            target["y"]
        )

        current_position = list(
            state.farmer_position
        )

        # -----------------------------------------------------
        # Verify that the target crop still exists.
        # -----------------------------------------------------

        target_crop = self._crop_at_target(
            state,
            target,
        )

        if target_crop is None:
            return ["PASS"]

        # -----------------------------------------------------
        # Restore planning target from persistent intent.
        # -----------------------------------------------------

        self.planning_target = PlanningTarget(
            x=target_x,
            y=target_y,
        )

        # -----------------------------------------------------
        # Farmer reached harvest target.
        # -----------------------------------------------------

        if current_position == self.planning_target.position:
            return ["HARVEST"]

        # -----------------------------------------------------
        # Navigate toward harvest target.
        # -----------------------------------------------------

        next_move = self.planner.next_move(
            current_position=current_position,
            target=self.planning_target,
        )

        if next_move is None:
            return ["HARVEST"]

        return [next_move]

    # =========================================================
    # CLEAR WEED
    # =========================================================

    def _clear_weed_action(
        self,
        decision_action,
        state: FarmState,
    ):
        """
        Navigate toward a weed-clearing target.

        CLEAR_WEED is a planning command.

        The environment receives DIG once the farmer reaches
        the target.
        """

        self.selected_spatial_target = None

        # -----------------------------------------------------
        # Persistent intent target.
        # -----------------------------------------------------

        if (
            self.execution_intent is not None
            and self.execution_intent.target is not None
        ):
            target_data = self.execution_intent.target

            self.planning_target = PlanningTarget(
                x=int(target_data["x"]),
                y=int(target_data["y"]),
            )

        # -----------------------------------------------------
        # Direct / legacy CLEAR_WEED command.
        # -----------------------------------------------------

        else:
            if len(decision_action) < 3:
                self.planning_target = None
                return ["PASS"]

            target_x = int(
                decision_action[1]
            )

            target_y = int(
                decision_action[2]
            )

            self.planning_target = PlanningTarget(
                x=target_x,
                y=target_y,
            )

        # -----------------------------------------------------
        # Safety check.
        # -----------------------------------------------------

        if self.planning_target is None:
            return ["PASS"]

        current_position = list(
            state.farmer_position
        )

        # -----------------------------------------------------
        # Farmer reached weed target.
        # -----------------------------------------------------

        if (
            current_position
            == self.planning_target.position
        ):
            return ["DIG"]

        # -----------------------------------------------------
        # Navigate toward weed target.
        # -----------------------------------------------------

        next_move = self.planner.next_move(
            current_position=current_position,
            target=self.planning_target,
        )

        if next_move is None:
            return ["DIG"]

        return [next_move]

    # =========================================================
    # PLANNING METADATA
    # =========================================================

    def _retrieval_target(
        self,
        decision_action,
        state: FarmState,
    ):
        """
        Return the spatial target representation used by
        experience retrieval.

        Retrieval is descriptive only; this helper does not
        select or modify a target.
        """

        if not decision_action:
            return None

        action_name = decision_action[0]

        if action_name in {
            "CLEAR_WEED",
            "PLANT_WHEAT",
            "PLANT_CARROT",
            "PLANT_TOMATO",
            "PLANT_STRAWBERRY",
            "PLANT_MELON",
        }:
            if len(decision_action) >= 3:
                return {
                    "x": int(decision_action[1]),
                    "y": int(decision_action[2]),
                }

        return None

    def _planning_metadata(
        self,
        decision_action,
        state: FarmState,
        farmer_action,
    ):
        """
        Record planning information for auditability.

        This method is used for non-persistent decisions.
        Persistent spatial decisions use _execution_metadata().
        """

        action_name = (
            decision_action[0]
            if decision_action
            else "PASS"
        )

        metadata = {
            "planning_target": None,
            "planned_move": None,
            "spatial_target": None,
            "spatial_target_reason": None,
        }

        # -----------------------------------------------------
        # CLEAR_WEED
        # -----------------------------------------------------

        if action_name == "CLEAR_WEED":
            if len(decision_action) < 3:
                return metadata

            target = {
                "x": int(decision_action[1]),
                "y": int(decision_action[2]),
            }

            metadata["planning_target"] = target

            if farmer_action:
                metadata["planned_move"] = (
                    farmer_action[0]
                )

            return metadata

        # -----------------------------------------------------
        # HARVEST
        # -----------------------------------------------------

        if action_name == "HARVEST":
            if farmer_action:
                metadata["planned_move"] = (
                    farmer_action[0]
                )

            return metadata

        # -----------------------------------------------------
        # PLANTING
        # -----------------------------------------------------

        plant_intent_names = {
            "PLANT_WHEAT",
            "PLANT_CARROT",
            "PLANT_TOMATO",
            "PLANT_STRAWBERRY",
            "PLANT_MELON",
        }

        is_plant_intent = (
            action_name in plant_intent_names
            or (
                action_name == "PLANT"
                and len(decision_action) >= 2
                and decision_action[1] in {
                    "WHEAT",
                    "CARROT",
                    "TOMATO",
                    "STRAWBERRY",
                    "MELON",
                }
            )
        )

        if is_plant_intent:
            if self.selected_spatial_target is None:
                return metadata

            target = self.selected_spatial_target

            metadata["spatial_target"] = {
                "x": target.x,
                "y": target.y,
            }

            metadata["planning_target"] = {
                "x": target.x,
                "y": target.y,
            }

            metadata["spatial_target_reason"] = (
                target.reason
            )

            if farmer_action:
                metadata["planned_move"] = (
                    farmer_action[0]
                )

            return metadata

        return metadata

    # =========================================================
    # MARKET ACTION
    # =========================================================

    @staticmethod
    def _market_action(action):
        """
        Extract market operations from a decision.

        Market operations and farmer operations are separated
        because Kaggriculture executes them in different
        action channels.
        """

        if not action:
            return []

        market_operations = {
            "BUY_SEED",
            "BUY_PRODUCT",
            "BUY_ANIMAL",
            "SELL",
            "HIRE",
            "BUY_LAND",
        }

        if action[0] in market_operations:
            return [action]

        return []