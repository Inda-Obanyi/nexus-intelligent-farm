from nexus.core.decision.context import DecisionContext
from nexus.core.decision.options import DecisionOption
from nexus.intelligence.prediction.models import Prediction
from nexus.intelligence.prediction.policy import (
    PredictionPolicy,
    PredictionPolicyEvaluator,
)


class DecisionScorer:
    """
    Scores candidate decisions using multiple NEXUS objectives.

    The scorer combines:

    - candidate base score
    - farm constraints
    - explicit management goals
    - crop-aware production utility
    - economic SELL utility
    - operational protection value
    - harvest execution utility
    - prediction-derived evidence

    Prediction evidence is used only when it satisfies an explicit
    PredictionPolicy.

    Architectural boundary:

        Farm State
            +
        Goals / Constraints
            +
        Risk Signals
            +
        Validated Predictions
            +
        Economic Signals
            +
        Execution Signals
            ↓
        DecisionScorer
            ↓
        DecisionEngine
    """

    def __init__(
        self,
        prediction_policy=None,
        prediction_evaluator=None,
    ):
        self.prediction_policy = (
            prediction_policy
            or PredictionPolicy()
        )

        self.prediction_evaluator = (
            prediction_evaluator
            or PredictionPolicyEvaluator()
        )

    def score(
        self,
        option: DecisionOption,
        context: DecisionContext,
    ) -> float:
        """
        Calculate the total score for a candidate decision.

        Prediction evidence contributes only when the relevant
        prediction exists and satisfies the configured policy.

        SELL decisions additionally receive bounded economic
        utility based on:

        - realizable sale value,
        - inventory pressure,
        - liquidity pressure.

        HARVEST decisions additionally receive bounded execution
        utility based on whether:

        - a harvestable crop exists anywhere on the farm;
        - the farmer is already standing on a harvestable crop.

        The local execution bonus is intentionally stronger than
        the distant-farm bonus.
        """

        score = 0.0

        if context.state.resources.money > 0:
            score += option.score

        if (
            context.state.resources.money
            <= context.constraints.minimum_cash
            and option.name.startswith("BUY_")
        ):
            score -= 5.0

        score += self._goal_score(
            option,
            context,
        )

        score += self._crop_production_score(
            option,
            context,
        )

        score += self._sell_economic_score(
            option,
            context,
        )

        score += self._harvest_execution_score(
            option,
            context,
        )

        if option.name == "WATER_CROP":
            score += 3.0

        score += self._risk_score(
            option,
            context,
        )

        score += self._prediction_score(
            option,
            context,
        )

        return score

    def _risk_score(
        self,
        option,
        context,
    ):
        """
        Apply risk-aware scoring when LOW_RISK_ONLY is enabled.
        """

        if "LOW_RISK_ONLY" not in context.constraints.allowed_actions:
            return 0.0

        if not option.name.startswith("PLANT_"):
            return 0.0

        crop_name = option.name[
            len("PLANT_"):
        ].upper()

        risk = context.crop_risks.get(
            crop_name
        )

        if risk is None:
            return 0.0

        weighted_risk = risk.weighted_overall(
            context.risk_weights
        )

        return -5.0 * weighted_risk

    def _goal_score(
        self,
        option,
        context,
    ):
        """
        Add score for explicit management goals.
        """

        score = 0.0

        for goal in context.goals:
            prefix = "prioritize_"

            if not goal.name.startswith(prefix):
                continue

            crop_name = goal.name[
                len(prefix):
            ].upper()

            if option.name in {
                f"BUY_{crop_name}_SEED",
                f"PLANT_{crop_name}",
            }:
                score += (
                    10.0
                    * goal.priority
                )

        return score

    def _crop_production_score(
        self,
        option,
        context,
    ):
        """
        Score planting options using production economics.
        """

        if not option.name.startswith("PLANT_"):
            return 0.0

        crop_name = option.name[
            len("PLANT_"):
        ].upper()

        try:
            crop_definition = (
                context.crop_definition(
                    crop_name
                )
            )
        except ValueError:
            return 0.0

        market_price = context.market_price(
            crop_name
        )

        if market_price <= 0:
            return 0.0

        expected_gross_value = (
            market_price
            * crop_definition.max_yield
        )

        expected_margin = (
            expected_gross_value
            - crop_definition.seed_cost
        )

        maturity_days = max(
            crop_definition.first_yield_day,
            1,
        )

        time_adjusted_margin = (
            expected_margin
            / maturity_days
        )

        return max(
            time_adjusted_margin,
            0.0,
        ) / 100.0

    def _sell_economic_score(
        self,
        option,
        context,
    ):
        """
        Add bounded economic utility to SELL decisions.
        """

        if not option.name.startswith("SELL_"):
            return 0.0

        action = list(option.action)

        if len(action) < 3:
            return 0.0

        if str(action[0]).upper() != "SELL":
            return 0.0

        product_name = str(action[1]).upper()

        try:
            quantity = float(action[2])
        except (TypeError, ValueError):
            return 0.0

        if quantity <= 0:
            return 0.0

        available_quantity = context.product_count(
            product_name
        )

        if available_quantity <= 0:
            return 0.0

        quantity = min(
            quantity,
            float(available_quantity),
        )

        market_price = context.market_price(
            product_name
        )

        if market_price <= 0:
            return 0.0

        sale_value = (
            quantity
            * market_price
        )

        sale_value_score = min(
            sale_value / 25.0,
            8.0,
        )

        inventory_score = min(
            float(available_quantity) * 0.5,
            3.0,
        )

        minimum_cash = float(
            context.constraints.minimum_cash
        )

        current_cash = float(
            context.state.resources.money
        )

        if minimum_cash > 0 and current_cash <= minimum_cash:
            liquidity_score = 5.0
        elif minimum_cash > 0:
            cash_buffer = (
                current_cash
                - minimum_cash
            )

            liquidity_score = max(
                0.0,
                5.0
                * (
                    1.0
                    - min(
                        cash_buffer
                        / max(
                            minimum_cash,
                            1.0,
                        ),
                        1.0,
                    )
                ),
            )
        else:
            liquidity_score = 0.0

        return (
            sale_value_score
            + inventory_score
            + liquidity_score
        )

    def _harvest_execution_score(
        self,
        option,
        context,
    ):
        """
        Reward harvest decisions using two execution signals.

        1. Farm-level harvest signal: +1

           At least one mature crop exists somewhere on the farm.

           This allows the DecisionEngine to choose HARVEST_CROP
           even when the farmer must first navigate to the crop.

        2. Immediate execution signal: +2

           The farmer is already standing on a mature crop.

           This stronger signal rewards completing an action that
           requires no additional navigation.

        Total possible harvest contribution:

            +3

        This remains bounded and therefore does not turn harvesting
        into an unconditional rule.
        """

        if option.name != "HARVEST_CROP":
            return 0.0

        harvestable_crops = (
            context.harvestable_crops()
        )

        if not harvestable_crops:
            return 0.0

        score = 1.0

        if self._can_harvest_current_crop(
            context
        ):
            score += 2.0

        return score

    @staticmethod
    def _can_harvest_current_crop(
        context,
    ):
        """
        Validate that the farmer is currently standing on a
        harvestable crop.
        """

        farmer_x, farmer_y = (
            context.state.farmer_position
        )

        current_crop = next(
            (
                crop
                for crop in context.state.crops
                if crop.row == farmer_y
                and crop.col == farmer_x
            ),
            None,
        )

        if current_crop is None:
            return False

        return context.is_crop_harvestable(
            current_crop
        )

    def _prediction_score(
        self,
        option,
        context,
    ):
        """
        Add interpretable decision value from validated predictions.

        Current supported prediction:

            expected_yield
        """

        if not option.name.startswith("PLANT_"):
            return 0.0

        crop_name = option.name[
            len("PLANT_"):
        ].upper()

        prediction = context.get_prediction(
            "expected_yield"
        )

        if prediction is None:
            return 0.0

        validation = (
            self.prediction_evaluator.evaluate(
                prediction,
                self.prediction_policy,
            )
        )

        if not validation.accepted:
            return 0.0

        if not isinstance(
            prediction.value,
            (int, float),
        ):
            return 0.0

        predicted_crop = prediction.evidence.get(
            "crop"
        )

        if predicted_crop is not None:
            if (
                str(predicted_crop).upper()
                != crop_name
            ):
                return 0.0

        predicted_yield = float(
            prediction.value
        )

        if predicted_yield <= 0:
            return 0.0

        return self._scale_prediction_value(
            predicted_yield,
            prediction,
        )

    @staticmethod
    def _scale_prediction_value(
        predicted_yield,
        prediction: Prediction,
    ):
        """
        Convert an expected-yield prediction into a modest
        interpretable score contribution.
        """

        confidence_factor = prediction.confidence

        uncertainty_factor = (
            1.0
            - prediction.uncertainty
        )

        reliability_factor = (
            confidence_factor
            * uncertainty_factor
        )

        return min(
            predicted_yield,
            10.0,
        ) * reliability_factor
