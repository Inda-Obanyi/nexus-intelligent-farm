from typing import List, Optional

from nexus.core.decision.context import DecisionContext
from nexus.core.decision.options import DecisionOption
from nexus.core.decision.preconditions import ActionPreconditions
from nexus.core.decision.scoring import DecisionScorer
from nexus.core.decision.trace import CandidateScore, DecisionTrace


class DecisionEngine:
    """
    Core NEXUS decision engine.

    Responsibilities:
    1. Generate candidate actions and goals.
    2. Ask the DecisionScorer to evaluate them.
    3. Preserve the complete scored candidate landscape.
    4. Build a DecisionTrace for decision-quality analysis.
    5. Select the highest-scoring valid option.

    The decision engine does not execute movement plans.
    Navigation is handled by the planning layer.

    Public compatibility:
        decide(context) still returns a DecisionOption.

    Decision-quality information:
        last_trace contains the scored candidates, selected decision,
        runner-up, score margin, candidate count, and interpretability
        confidence label.

    Architectural boundary:

        DecisionEngine
            WHAT should NEXUS do?

        SpatialTargetSelector
            WHERE should NEXUS do it?

        FarmPlanner
            HOW should NEXUS get there?
    """

    def __init__(self):
        self.scorer = DecisionScorer()
        self.preconditions = ActionPreconditions()
        self.last_trace: Optional[DecisionTrace] = None

    def decide(
        self,
        context: DecisionContext,
    ) -> DecisionOption:
        """
        Generate, score, trace, and select the best valid decision.
        """

        options = self._generate_options(context)

        if not options:
            decision = DecisionOption(
                name="PASS",
                action=["PASS"],
                score=0.0,
                reason="No valid action identified.",
            )

            self.last_trace = DecisionTrace.from_scored_candidates(
                [
                    CandidateScore(
                        name=decision.name,
                        score=decision.score,
                        action=decision.action,
                    )
                ]
            )

            return decision

        scored_options = []

        for option in options:
            scored_score = self.scorer.score(
                option,
                context,
            )

            scored_options.append(
                DecisionOption(
                    name=option.name,
                    action=option.action,
                    score=scored_score,
                    reason=option.reason,
                )
            )

        self.last_trace = DecisionTrace.from_scored_candidates(
            [
                CandidateScore(
                    name=option.name,
                    score=option.score,
                    action=option.action,
                )
                for option in scored_options
            ]
        )

        return max(
            scored_options,
            key=lambda option: option.score,
        )

    def _has_prioritized_crop(
        self,
        context: DecisionContext,
        crop_name: str,
    ) -> bool:
        """
        Return whether a crop has an explicit management-priority goal.

        Goals use the convention:

            prioritize_<crop>

        Example:

            prioritize_carrot

        The decision engine uses this only to determine whether a
        missing seed should be considered for targeted acquisition.
        The scorer remains responsible for applying the actual
        priority weight.
        """

        target = crop_name.lower()

        return any(
            goal.name.lower() == f"prioritize_{target}"
            for goal in context.goals
        )

    def _generate_options(
        self,
        context: DecisionContext,
    ) -> List[DecisionOption]:
        """
        Generate all currently relevant decision candidates.

        The engine determines WHAT should happen.

        It does not determine HOW the farmer navigates to a
        spatial target. That responsibility remains with the
        planning layer.
        """

        options = []

        from nexus.intelligence.crops.definitions import (
            CROP_DEFINITIONS,
        )

        # ---------------------------------------------------------
        # CROP-SPECIFIC SEED ACQUISITION
        # ---------------------------------------------------------

        has_any_available_seed = any(
            count > 0
            for count in context.state.resources.seeds.values()
        )

        for crop_name, crop_definition in CROP_DEFINITIONS.items():
            targeted_acquisition = self._has_prioritized_crop(
                context,
                crop_name,
            )

            should_consider_purchase = (
                not has_any_available_seed
                or targeted_acquisition
            )

            if not should_consider_purchase:
                continue

            buy_seed_check = self.preconditions.can_buy_seed(
                context,
                crop_name,
            )

            if buy_seed_check.valid:
                if targeted_acquisition:
                    acquisition_reason = (
                        f"{crop_name.title()} seed costs "
                        f"{crop_definition.seed_cost} and is being "
                        "considered because the crop has an explicit "
                        "management-priority goal."
                    )
                else:
                    acquisition_reason = (
                        f"{crop_name.title()} seed costs "
                        f"{crop_definition.seed_cost} and enables "
                        "crop production."
                    )

                options.append(
                    DecisionOption(
                        name=f"BUY_{crop_name}_SEED",
                        action=[
                            "BUY_SEED",
                            crop_name,
                            1,
                        ],
                        score=10.0,
                        reason=acquisition_reason,
                    )
                )

        # ---------------------------------------------------------
        # SPATIAL PLANTING FEASIBILITY
        # ---------------------------------------------------------

        has_plantable_tile = bool(
            context.state.plantable_tiles()
        )

        for crop_name in CROP_DEFINITIONS:
            plant_check = self.preconditions.can_plant_crop(
                context,
                crop_name,
            )

            if plant_check.valid:
                options.append(
                    DecisionOption(
                        name=f"PLANT_{crop_name}",
                        action=[
                            "PLANT",
                            crop_name,
                        ],
                        score=8.0,
                        reason=plant_check.reason,
                    )
                )
                continue

            if (
                has_plantable_tile
                and context.seed_count(crop_name) > 0
            ):
                options.append(
                    DecisionOption(
                        name=f"PLANT_{crop_name}",
                        action=[
                            "PLANT",
                            crop_name,
                        ],
                        score=8.0,
                        reason=(
                            f"{crop_name.title()} seed is available "
                            "and the farm has at least one plantable "
                            "tile. Spatial planning should select "
                            "the planting location before execution."
                        ),
                    )
                )

        # ---------------------------------------------------------
        # WATERING
        # ---------------------------------------------------------

        water_check = self.preconditions.can_water_crop(
            context
        )

        if water_check.valid:
            options.append(
                DecisionOption(
                    name="WATER_CROP",
                    action=["WATER"],
                    score=9.0,
                    reason=water_check.reason,
                )
            )

        # ---------------------------------------------------------
        # FARM-LEVEL HARVESTING
        # ---------------------------------------------------------
        #
        # IMPORTANT:
        #
        # Harvest generation is deliberately FARM-LEVEL.
        #
        # The DecisionEngine answers:
        #
        #     "Is there a harvestable crop anywhere on the farm?"
        #
        # It does NOT require the farmer to already be standing
        # on that crop.
        #
        # SpatialTargetSelector later answers:
        #
        #     "Which harvestable crop should we move toward?"
        #
        # FarmPlanner then answers:
        #
        #     "How do we get there?"
        #
        # This separation allows:
        #
        #     Farmer at [4,4]
        #     Crop at [4,7]
        #
        # to correctly produce:
        #
        #     HARVEST_CROP
        #
        # rather than waiting until the farmer accidentally reaches
        # the crop.
        # ---------------------------------------------------------

        harvestable_crops = context.harvestable_crops()

        if harvestable_crops:
            crop_summary = ", ".join(
                (
                    f"{crop.crop} at "
                    f"[{crop.col}, {crop.row}] "
                    f"({crop.yield_units} units)"
                )
                for crop in harvestable_crops
            )

            options.append(
                DecisionOption(
                    name="HARVEST_CROP",
                    action=["HARVEST"],
                    score=12.0,
                    reason=(
                        f"{len(harvestable_crops)} harvestable crop(s) "
                        f"detected on the farm: {crop_summary}. "
                        "Spatial planning should select the nearest "
                        "harvestable crop and move the farmer there "
                        "before harvesting."
                    ),
                )
            )

        # ---------------------------------------------------------
        # PRODUCT SELLING
        # ---------------------------------------------------------

        for product_name in CROP_DEFINITIONS:
            sell_check = self.preconditions.can_sell_product(
                context,
                product_name,
            )

            if not sell_check.valid:
                continue

            quantity = context.product_count(
                product_name
            )

            options.append(
                DecisionOption(
                    name=f"SELL_{product_name}",
                    action=[
                        "SELL",
                        product_name,
                        quantity,
                    ],
                    score=13.0,
                    reason=sell_check.reason,
                )
            )

        # ---------------------------------------------------------
        # LOCAL WEED MANAGEMENT
        # ---------------------------------------------------------

        dig_weed_check = self.preconditions.can_dig_weed(
            context
        )

        if dig_weed_check.valid:
            options.append(
                DecisionOption(
                    name="DIG_WEED",
                    action=["DIG"],
                    score=16.0,
                    reason=(
                        f"{dig_weed_check.reason} "
                        "Immediate environmental maintenance "
                        "takes priority over acquiring new resources."
                    ),
                )
            )

        # ---------------------------------------------------------
        # DISTANT WEED MANAGEMENT
        # ---------------------------------------------------------

        farmer_x, farmer_y = context.state.farmer_position

        distant_weeds = [
            weed
            for weed in context.state.weeds
            if weed.col != farmer_x
            or weed.row != farmer_y
        ]

        if distant_weeds:
            target_weed = min(
                distant_weeds,
                key=lambda weed: (
                    abs(weed.col - farmer_x)
                    + abs(weed.row - farmer_y),
                    weed.row,
                    weed.col,
                ),
            )

            target_x = target_weed.col
            target_y = target_weed.row

            distance = (
                abs(target_x - farmer_x)
                + abs(target_y - farmer_y)
            )

            options.append(
                DecisionOption(
                    name="CLEAR_WEED",
                    action=[
                        "CLEAR_WEED",
                        target_x,
                        target_y,
                    ],
                    score=11.0,
                    reason=(
                        f"Weed detected at [{target_x}, {target_y}]. "
                        f"The farmer is at [{farmer_x}, {farmer_y}], "
                        f"{distance} movement steps away. "
                        "Navigation planning should move toward "
                        "the weed before clearing it."
                    ),
                )
            )

        # ---------------------------------------------------------
        # PASS
        # ---------------------------------------------------------

        options.append(
            DecisionOption(
                name="PASS",
                action=["PASS"],
                score=0.0,
                reason="No higher-priority action selected.",
            )
        )

        return options
