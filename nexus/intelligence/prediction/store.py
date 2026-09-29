from typing import Dict, List, Optional

from nexus.intelligence.prediction.models import Prediction


class PredictionStore:
    """
    In-memory repository for NEXUS predictions.

    Responsibilities:

    - store predictions
    - retrieve predictions by target
    - retrieve all predictions
    - replace the latest prediction for a target
    - clear stored predictions

    The store does not evaluate prediction quality and does not
    make farm decisions.

    Architectural boundary:

        Intelligence Components
                 ↓
            Prediction
                 ↓
          PredictionStore
                 ↓
        Risk / Decision Context
    """

    def __init__(self):
        self._predictions: Dict[str, Prediction] = {}

    def save(self, prediction: Prediction) -> Prediction:
        """
        Store a prediction as the latest prediction for its target.

        If another prediction already exists for the same target,
        it is replaced.

        Returns:
            The stored prediction.
        """

        target = prediction.target.strip()

        self._predictions[target] = prediction

        return prediction

    def get(
        self,
        target: str,
    ) -> Optional[Prediction]:
        """
        Retrieve the latest prediction for a target.

        Target lookup is normalized by trimming surrounding
        whitespace.
        """

        return self._predictions.get(
            target.strip()
        )

    def get_all(self) -> List[Prediction]:
        """
        Return all currently stored predictions.

        The returned list is a snapshot of the store and does not
        expose the internal dictionary.
        """

        return list(
            self._predictions.values()
        )

    def contains(self, target: str) -> bool:
        """
        Return whether a prediction exists for the target.
        """

        return target.strip() in self._predictions

    def count(self) -> int:
        """
        Return the number of stored prediction targets.
        """

        return len(
            self._predictions
        )

    def remove(
        self,
        target: str,
    ) -> Optional[Prediction]:
        """
        Remove and return the prediction for a target.

        Returns None when the target does not exist.
        """

        return self._predictions.pop(
            target.strip(),
            None,
        )

    def clear(self) -> None:
        """
        Remove all stored predictions.
        """

        self._predictions.clear()
