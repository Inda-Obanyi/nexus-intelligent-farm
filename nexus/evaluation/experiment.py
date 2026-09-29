from typing import Any, Dict, List

from nexus.agents.baseline.agent import BaselineAgent
from nexus.agents.nexus.runner import NexusAgentRunner
from nexus.environments.kaggriculture.environment import KaggricultureEnvironment
from nexus.evaluation.decision_quality import decision_quality_from_records
from nexus.evaluation.metrics import evaluate_episode


STARTING_MONEY = 3000.0


def _opponent_action() -> Dict[str, Any]:
    """Return a deterministic inactive opponent."""
    return {
        "farmer": ["PASS"],
        "hands": [],
        "market": [],
    }


def _run_baseline_episode(
    episode_steps: int,
    debug: bool,
    seed: int | None = None,
) -> Dict[str, float]:
    """Run one deterministic baseline episode."""

    agent = BaselineAgent()

    environment = KaggricultureEnvironment(
        episode_steps=episode_steps,
        debug=debug,
        seed=seed,
    )

    environment.reset()

    for _ in range(episode_steps):
        if environment.state[0].status != "ACTIVE":
            break

        observation = environment.state[0].observation
        action = agent.act(observation)

        environment.step(
            [
                action,
                _opponent_action(),
            ]
        )

    final_observation = environment.state[0].observation

    final_money = final_observation["farms"][0]["money"]
    shed = final_observation["private"]["shed"]

    return evaluate_episode(
        starting_money=STARTING_MONEY,
        final_money=final_money,
        shed=shed,
    )


def _summarize_decision_intelligence(
    records,
) -> Dict[str, float]:
    """
    Summarize decision-intelligence records.

    Canonical effect-match values are:

        MATCHED
        PARTIAL
        NOT_MATCHED
        NOT_APPLICABLE

    The legacy MISMATCH value is also accepted so older
    stored reports remain compatible.

    NOT_MATCHED and MISMATCH are both aggregated as
    mismatch outcomes.
    """

    intelligence_reports = []

    for record in records:
        outcome = getattr(
            record,
            "outcome",
            None,
        )

        if not outcome:
            continue

        intelligence = outcome.get(
            "decision_intelligence"
        )

        if intelligence:
            intelligence_reports.append(
                intelligence
            )

    if not intelligence_reports:
        return {
            "decision_intelligence_count": 0.0,
            "matched_effect_count": 0.0,
            "partial_effect_count": 0.0,
            "mismatch_effect_count": 0.0,
            "unknown_effect_count": 0.0,
            "effect_match_rate": 0.0,
            "mean_alternative_count": 0.0,
            "mean_score_margin": 0.0,
        }

    matched = sum(
        report.get("effect_match") == "MATCHED"
        for report in intelligence_reports
    )

    partial = sum(
        report.get("effect_match") == "PARTIAL"
        for report in intelligence_reports
    )

    mismatch = sum(
        report.get("effect_match")
        in {
            "NOT_MATCHED",
            "MISMATCH",
        }
        for report in intelligence_reports
    )

    unknown = sum(
        report.get("effect_match")
        in {
            None,
            "",
            "UNKNOWN",
        }
        for report in intelligence_reports
    )

    total = len(
        intelligence_reports
    )

    alternative_counts = [
        len(
            report.get(
                "alternatives",
                [],
            )
        )
        for report in intelligence_reports
    ]

    score_margins = [
        float(
            report.get(
                "score_margin",
                0.0,
            )
        )
        for report in intelligence_reports
    ]

    return {
        "decision_intelligence_count": float(
            total
        ),
        "matched_effect_count": float(
            matched
        ),
        "partial_effect_count": float(
            partial
        ),
        "mismatch_effect_count": float(
            mismatch
        ),
        "unknown_effect_count": float(
            unknown
        ),
        "effect_match_rate": (
            float(matched) / total
            if total
            else 0.0
        ),
        "mean_alternative_count": (
            sum(alternative_counts) / total
            if total
            else 0.0
        ),
        "mean_score_margin": (
            sum(score_margins) / total
            if total
            else 0.0
        ),
    }


def _run_nexus_episode(
    episode_steps: int,
    debug: bool,
    seed: int | None = None,
) -> Dict[str, float]:
    """Run one NEXUS episode and collect evaluation evidence."""

    runner = NexusAgentRunner(
        episode_steps=episode_steps,
        debug=debug,
        seed=seed,
    )

    episode = runner.run()

    result = dict(
        episode["evaluation"]
    )

    records = runner.agent.memory.all()

    quality = decision_quality_from_records(
        records
    )

    result.update(
        {
            "decision_count": float(
                quality.total_decisions
            ),
            "high_confidence_count": float(
                quality.high_confidence_count
            ),
            "medium_confidence_count": float(
                quality.medium_confidence_count
            ),
            "low_confidence_count": float(
                quality.low_confidence_count
            ),
            "average_score_margin": float(
                quality.average_score_margin
            ),
            "minimum_score_margin": float(
                quality.minimum_score_margin
            ),
            "maximum_score_margin": float(
                quality.maximum_score_margin
            ),
            "ambiguous_decision_rate": float(
                quality.ambiguous_decision_rate
            ),
            "average_candidate_count": float(
                quality.average_candidate_count
            ),
        }
    )

    result.update(
        _summarize_decision_intelligence(
            records
        )
    )

    return result


def run_experiment(
    episodes: int = 5,
    episode_steps: int = 720,
    debug: bool = False,
    seeds: List[int] | None = None,
) -> Dict[str, List[Dict[str, float]]]:
    """
    Compare the baseline agent with NEXUS.

    When explicit seeds are supplied, each baseline and NEXUS
    episode receives the same environment seed so their
    conditions are directly comparable.
    """

    if episodes <= 0:
        raise ValueError(
            "episodes must be greater than zero."
        )

    if episode_steps <= 0:
        raise ValueError(
            "episode_steps must be greater than zero."
        )

    if seeds is not None:
        if len(seeds) != episodes:
            raise ValueError(
                "seeds must contain exactly one seed per episode."
            )

    baseline_results = []
    nexus_results = []

    for episode_index in range(
        episodes
    ):
        seed = (
            seeds[episode_index]
            if seeds is not None
            else None
        )

        baseline_results.append(
            _run_baseline_episode(
                episode_steps=episode_steps,
                debug=debug,
                seed=seed,
            )
        )

        nexus_results.append(
            _run_nexus_episode(
                episode_steps=episode_steps,
                debug=debug,
                seed=seed,
            )
        )

    return {
        "baseline": baseline_results,
        "nexus": nexus_results,
    }


def _summarize_agent_results(
    results: List[Dict[str, float]],
) -> Dict[str, float]:
    """Calculate descriptive statistics for one agent."""

    profits = [
        float(result["net_profit"])
        for result in results
    ]

    roi_values = [
        float(result["roi_percent"])
        for result in results
    ]

    inventory_values = [
        float(result["inventory_value"])
        for result in results
    ]

    def mean(values):
        return (
            sum(values) / len(values)
            if values
            else 0.0
        )

    def median(values):
        if not values:
            return 0.0

        ordered = sorted(values)
        middle = len(ordered) // 2

        if len(ordered) % 2 == 1:
            return ordered[middle]

        return (
            ordered[middle - 1]
            + ordered[middle]
        ) / 2.0

    def standard_deviation(values):
        if not values:
            return 0.0

        average = mean(values)

        variance = sum(
            (value - average) ** 2
            for value in values
        ) / len(values)

        return variance ** 0.5

    return {
        "mean_profit": mean(profits),
        "median_profit": median(profits),
        "std_profit": standard_deviation(
            profits
        ),
        "min_profit": min(profits)
        if profits
        else 0.0,
        "max_profit": max(profits)
        if profits
        else 0.0,
        "mean_roi_percent": mean(
            roi_values
        ),
        "median_roi_percent": median(
            roi_values
        ),
        "std_roi_percent": standard_deviation(
            roi_values
        ),
        "mean_inventory_value": mean(
            inventory_values
        ),
    }


def _has_decision_quality_metrics(
    result: Dict[str, float],
) -> bool:
    """Return whether a result contains decision-quality metrics."""

    required_fields = {
        "decision_count",
        "high_confidence_count",
        "medium_confidence_count",
        "low_confidence_count",
        "average_score_margin",
        "minimum_score_margin",
        "maximum_score_margin",
        "ambiguous_decision_rate",
        "average_candidate_count",
    }

    return required_fields.issubset(
        result
    )


def _summarize_nexus_decision_quality(
    results: List[Dict[str, float]],
) -> Dict[str, float]:
    """Summarize decision-quality metrics across NEXUS episodes."""

    if not results:
        return {}

    def mean(field):
        values = [
            float(result[field])
            for result in results
        ]

        return (
            sum(values) / len(values)
            if values
            else 0.0
        )

    return {
        "mean_decision_count": mean(
            "decision_count"
        ),
        "mean_high_confidence_count": mean(
            "high_confidence_count"
        ),
        "mean_medium_confidence_count": mean(
            "medium_confidence_count"
        ),
        "mean_low_confidence_count": mean(
            "low_confidence_count"
        ),
        "mean_score_margin": mean(
            "average_score_margin"
        ),
        "minimum_score_margin": min(
            float(result["minimum_score_margin"])
            for result in results
        ),
        "maximum_score_margin": max(
            float(result["maximum_score_margin"])
            for result in results
        ),
        "mean_ambiguous_decision_rate": mean(
            "ambiguous_decision_rate"
        ),
        "mean_candidate_count": mean(
            "average_candidate_count"
        ),
    }


def _has_decision_intelligence_metrics(
    result: Dict[str, float],
) -> bool:
    """Return whether a result contains decision-intelligence metrics."""

    required_fields = {
        "decision_intelligence_count",
        "matched_effect_count",
        "partial_effect_count",
        "mismatch_effect_count",
        "effect_match_rate",
        "mean_alternative_count",
        "mean_score_margin",
    }

    return required_fields.issubset(
        result
    )


def _summarize_nexus_decision_intelligence(
    results: List[Dict[str, float]],
) -> Dict[str, float]:
    """Summarize decision-intelligence metrics across NEXUS episodes."""

    if not results:
        return {}

    def mean(field):
        values = [
            float(
                result.get(
                    field,
                    0.0,
                )
            )
            for result in results
        ]

        return (
            sum(values) / len(values)
            if values
            else 0.0
        )

    return {
        "mean_decision_intelligence_count": mean(
            "decision_intelligence_count"
        ),
        "mean_matched_effect_count": mean(
            "matched_effect_count"
        ),
        "mean_partial_effect_count": mean(
            "partial_effect_count"
        ),
        "mean_mismatch_effect_count": mean(
            "mismatch_effect_count"
        ),
        "mean_unknown_effect_count": mean(
            "unknown_effect_count"
        ),
        "mean_effect_match_rate": mean(
            "effect_match_rate"
        ),
        "mean_alternative_count": mean(
            "mean_alternative_count"
        ),
        "mean_score_margin": mean(
            "mean_score_margin"
        ),
    }


def summarize_experiment(
    report: Dict[str, List[Dict[str, float]]],
) -> Dict[str, Any]:
    """
    Produce descriptive experiment statistics.

    These statistics describe observed differences under the
    supplied experiment conditions. They do not establish
    statistical significance, causal superiority, or global
    optimality.
    """

    baseline_results = report.get(
        "baseline",
        []
    )

    nexus_results = report.get(
        "nexus",
        []
    )

    baseline_summary = _summarize_agent_results(
        baseline_results
    )

    nexus_summary = _summarize_agent_results(
        nexus_results
    )

    baseline_mean_profit = baseline_summary[
        "mean_profit"
    ]

    nexus_mean_profit = nexus_summary[
        "mean_profit"
    ]

    profit_improvement = (
        nexus_mean_profit
        - baseline_mean_profit
    )

    if baseline_mean_profit != 0:
        profit_improvement_percent = (
            profit_improvement
            / abs(baseline_mean_profit)
            * 100.0
        )
    else:
        profit_improvement_percent = 0.0

    roi_difference = (
        nexus_summary["mean_roi_percent"]
        - baseline_summary["mean_roi_percent"]
    )

    inventory_improvement = (
        nexus_summary["mean_inventory_value"]
        - baseline_summary[
            "mean_inventory_value"
        ]
    )

    summary = {
        "baseline": baseline_summary,
        "nexus": nexus_summary,
        "comparison": {
            "profit_improvement": profit_improvement,
            "profit_improvement_percent": profit_improvement_percent,
            "roi_difference_percentage_points": roi_difference,
            "inventory_value_improvement": inventory_improvement,
        },
    }

    if nexus_results and all(
        _has_decision_quality_metrics(
            result
        )
        for result in nexus_results
    ):
        summary["nexus_decision_quality"] = (
            _summarize_nexus_decision_quality(
                nexus_results
            )
        )

    if nexus_results and all(
        _has_decision_intelligence_metrics(
            result
        )
        for result in nexus_results
    ):
        summary[
            "nexus_decision_intelligence"
        ] = _summarize_nexus_decision_intelligence(
            nexus_results
        )

    return summary
