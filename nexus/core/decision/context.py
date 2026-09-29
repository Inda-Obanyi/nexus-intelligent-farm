from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nexus.core.memory.learning_signal import LearningSignal
from nexus.core.state import FarmState
from nexus.intelligence.crops.definitions import (
    CropDefinition,
    get_crop_definition,
)
from nexus.intelligence.prediction.models import Prediction
from nexus.intelligence.risk.models import CropRisk


DEFAULT_RISK_WEIGHTS = {
    "weather": 0.20,
    "disease": 0.20,
    "pest": 0.20,
    "market": 0.20,
    "operational": 0.20,
}


@dataclass(frozen=True)
class RiskPolicy:
    """Named policy defining how NEXUS weights crop-risk dimensions."""

    name: str
    weights: Dict[str, float]

    def __post_init__(self):
        expected_fields = {
            "weather",
            "disease",
            "pest",
            "market",
            "operational",
        }

        if set(self.weights) != expected_fields:
            raise ValueError(
                "Risk policy weights must contain exactly: "
                "weather, disease, pest, market, operational."
            )

        if any(weight < 0 for weight in self.weights.values()):
            raise ValueError(
                "Risk policy weights cannot be negative."
            )

        if sum(self.weights.values()) <= 0:
            raise ValueError(
                "Risk policy total weight must be greater than zero."
            )


@dataclass
class DecisionGoal:
    """Defines what the NEXUS decision system is trying to optimize."""

    name: str
    priority: float = 1.0


@dataclass
class DecisionConstraints:
    """Constraints that must be respected when making a decision."""

    minimum_cash: float = 0.0
    allowed_actions: List[str] = field(default_factory=list)


@dataclass
class DecisionContext:
    """
    NEXUS decision-making context built from the current farm state.

    The context combines:

    - observable farm state
    - management goals
    - operational constraints
    - structured crop-risk information
    - validated intelligence predictions

    Predictions are stored as intelligence evidence. They do not
    automatically become decisions.

    Architectural boundary:

        Farm State
            +
        Predictions
            +
        Risks
            +
        Goals / Constraints
            ↓
        DecisionContext
            ↓
        DecisionEngine
    """

    state: FarmState

    goals: List[DecisionGoal] = field(
        default_factory=list
    )

    constraints: DecisionConstraints = field(
        default_factory=DecisionConstraints
    )

    crop_risks: Dict[str, CropRisk] = field(
        default_factory=dict
    )

    predictions: Dict[str, Prediction] = field(
        default_factory=dict
    )

    risk_weights: Dict[str, float] = field(
        default_factory=lambda: DEFAULT_RISK_WEIGHTS.copy()
    )

    learning_signal: Optional[LearningSignal] = None

    def risk_score(self, crop_name: str) -> float:
        """
        Return the overall risk score for a crop.

        Unknown crops have no registered risk and therefore return
        zero rather than inventing a risk signal.
        """

        risk = self.crop_risks.get(
            crop_name.upper()
        )

        if risk is None:
            return 0.0

        return risk.overall

    @property
    def risk_scores(self) -> Dict[str, float]:
        """Return overall risk scores for all registered crops."""

        return {
            crop_name: risk.overall
            for crop_name, risk in self.crop_risks.items()
        }

    def set_risk_score(
        self,
        crop_name: str,
        score: float,
    ) -> None:
        """
        Register a simple weather-based risk score.

        This method is retained for backward compatibility with
        the existing decision system.
        """

        crop_name = crop_name.upper()

        self.crop_risks[crop_name] = CropRisk(
            crop=crop_name,
            weather=score,
        )

    def has_crop(self, crop_name: str) -> bool:
        """Return whether the farm currently contains the crop."""

        return any(
            crop.crop == crop_name
            for crop in self.state.crops
        )

    def seed_count(self, crop_name: str) -> int:
        """Return the number of seeds currently available."""

        return self.state.resources.seeds.get(
            crop_name,
            0,
        )

    def crop_definition(
        self,
        crop_name: str,
    ) -> CropDefinition:
        """Return the agricultural definition of a crop."""

        return get_crop_definition(
            crop_name
        )

    def is_crop_harvestable(
        self,
        crop,
    ) -> bool:

        crop_definition = self.crop_definition(
            crop.crop
        )

        crop_age_days = (
            self.state.day
            - crop.planted_day
        )

        return (
            crop.yield_units > 0
            and crop_age_days
            >= crop_definition.first_yield_day
        )

    def harvestable_crops(self) -> List[Any]:
     return [
        crop
        for crop in self.state.crops
        if self.is_crop_harvestable(crop)
    ]

    def product_count(
        self,
        product_name: str,
    ) -> int:
        """Return the quantity of a product in the farm shed."""

        return self.state.resources.shed.get(
            product_name,
            0,
        )

    def market_price(
        self,
        item: str,
    ) -> float:
        """Return the current market price for an item."""

        return self.state.market.prices.get(
            item,
            0.0,
        )

    def set_prediction(
        self,
        prediction: Prediction,
    ) -> None:
        """
        Register a prediction by its target.

        The prediction is stored as intelligence evidence.
        This method does not evaluate the prediction's quality
        and does not make a decision from it.

        Prediction validation remains the responsibility of
        PredictionPolicyEvaluator.
        """

        target = prediction.target.strip()

        self.predictions[target] = prediction

    def get_prediction(
        self,
        target: str,
    ) -> Optional[Prediction]:
        """
        Return the latest prediction for a target.

        Target lookup is normalized by trimming surrounding
        whitespace.
        """

        return self.predictions.get(
            target.strip()
        )

    def has_prediction(
        self,
        target: str,
    ) -> bool:
        """Return whether a prediction exists for a target."""

        return target.strip() in self.predictions

    @property
    def prediction_targets(self) -> List[str]:
        """
        Return the targets currently represented in the context.
        """

        return list(
            self.predictions.keys()
        )
