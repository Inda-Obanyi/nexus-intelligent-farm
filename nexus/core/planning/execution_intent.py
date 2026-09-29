from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExecutionIntent:
    """
    Persistent execution state for a strategic decision.

    The DecisionEngine determines WHAT should happen.

    ExecutionIntent persists that objective while the planner
    determines HOW to accomplish it over multiple environment
    timesteps.

    Lifecycle:

        IN_PROGRESS
            ├── objective achieved   → COMPLETED
            ├── objective impossible → FAILED
            └── otherwise            → IN_PROGRESS

    COMPLETED and FAILED are terminal states.
    """

    decision_name: str
    action: List[Any]
    target: Optional[Dict[str, int]] = None

    status: str = "IN_PROGRESS"

    steps: List[List[Any]] = field(
        default_factory=list
    )

    initial_crop_count: Optional[int] = None
    initial_weed_count: Optional[int] = None
    initial_seed_count: Optional[int] = None

    execution_steps: int = 0

    def record_step(
        self,
        action: List[Any],
    ) -> None:
        """
        Record one low-level action taken while executing
        this strategic intent.
        """
        self.steps.append(
            list(action)
        )

        self.execution_steps += 1

    def complete(self) -> None:
        """
        Mark the execution intent as completed.

        Completion is terminal, so a failed or already completed
        intent cannot be changed into another state.
        """
        if self.status == "IN_PROGRESS":
            self.status = "COMPLETED"

    def fail(self) -> None:
        """
        Mark the execution intent as failed.

        Failure is terminal, so a completed or already failed
        intent cannot be changed into another state.
        """
        if self.status == "IN_PROGRESS":
            self.status = "FAILED"

    def is_active(self) -> bool:
        """
        Return True while the intent still requires execution.
        """
        return self.status == "IN_PROGRESS"

    def is_completed(self) -> bool:
        """
        Return True when the strategic objective was achieved.
        """
        return self.status == "COMPLETED"

    def is_failed(self) -> bool:
        """
        Return True when the strategic objective could not be achieved.
        """
        return self.status == "FAILED"

    def as_dict(self) -> Dict[str, Any]:
        """
        Return a serializable representation of the execution intent.

        This is used by NEXUS decision/execution metadata and
        audit records.
        """
        return {
            "decision_name": self.decision_name,
            "action": list(self.action),
            "target": (
                dict(self.target)
                if self.target is not None
                else None
            ),
            "status": self.status,
            "steps": [
                list(step)
                for step in self.steps
            ],
            "initial_crop_count": self.initial_crop_count,
            "initial_weed_count": self.initial_weed_count,
            "initial_seed_count": self.initial_seed_count,
            "execution_steps": self.execution_steps,
        }
