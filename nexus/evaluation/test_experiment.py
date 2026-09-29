from types import SimpleNamespace

from nexus.evaluation.experiment import (
    _summarize_decision_intelligence,
    run_experiment,
    summarize_experiment,
)


def make_intelligence_record(effect_match):
    """Create the smallest record shape required by the summarizer."""
    return SimpleNamespace(
        outcome={
            "decision_intelligence": {
                "effect_match": effect_match,
                "alternatives": [
                    {"name": "PASS"},
                    {"name": "WATER_CROP"},
                ],
                "score_margin": 2.0,
            }
        }
    )


def test_experiment_compares_baseline_and_nexus():
    report = run_experiment(
        episodes=2,
        episode_steps=24,
        debug=False,
    )

    assert "baseline" in report
    assert "nexus" in report

    assert len(report["baseline"]) == 2
    assert len(report["nexus"]) == 2

    for result in report["baseline"] + report["nexus"]:
        assert "final_money" in result
        assert "inventory_value" in result
        assert "net_profit" in result
        assert "roi_percent" in result


def test_experiment_contains_comparable_metrics():
    report = run_experiment(
        episodes=1,
        episode_steps=10,
        debug=False,
    )

    baseline = report["baseline"][0]
    nexus = report["nexus"][0]

    assert baseline["starting_money"] == 3000.0
    assert nexus["starting_money"] == 3000.0


def test_experiment_summary_calculates_statistics():
    report = {
        "baseline": [
            {
                "starting_money": 3000.0,
                "final_money": 2820.0,
                "inventory_value": 350.0,
                "net_profit": 170.0,
                "roi_percent": 5.6666666667,
            },
            {
                "starting_money": 3000.0,
                "final_money": 2830.0,
                "inventory_value": 350.0,
                "net_profit": 180.0,
                "roi_percent": 6.0,
            },
        ],
        "nexus": [
            {
                "starting_money": 3000.0,
                "final_money": 2760.0,
                "inventory_value": 840.0,
                "net_profit": 600.0,
                "roi_percent": 20.0,
            },
            {
                "starting_money": 3000.0,
                "final_money": 2760.0,
                "inventory_value": 840.0,
                "net_profit": 600.0,
                "roi_percent": 20.0,
            },
        ],
    }

    summary = summarize_experiment(report)

    assert "baseline" in summary
    assert "nexus" in summary
    assert "comparison" in summary

    assert summary["baseline"]["mean_profit"] == 175.0
    assert summary["nexus"]["mean_profit"] == 600.0
    assert summary["comparison"]["profit_improvement"] == 425.0

    assert (
        abs(
            summary["comparison"]["profit_improvement_percent"]
            - 242.85714285714286
        )
        < 1e-10
    )


def test_experiment_accepts_explicit_seeds():
    report = run_experiment(
        episodes=2,
        episode_steps=24,
        debug=False,
        seeds=[101, 202],
    )

    assert len(report["baseline"]) == 2
    assert len(report["nexus"]) == 2


def test_experiment_rejects_mismatched_seed_count():
    try:
        run_experiment(
            episodes=3,
            episode_steps=24,
            debug=False,
            seeds=[101, 202],
        )
    except ValueError as exc:
        assert (
            str(exc)
            == "seeds must contain exactly one seed per episode."
        )
    else:
        raise AssertionError(
            "Expected ValueError for mismatched seed count."
        )


def test_decision_intelligence_summary_counts_not_matched_as_mismatch():
    """
    Regression test for the canonical NEXUS outcome vocabulary.

    DecisionOutcomeEvaluator uses:

        MATCHED
        PARTIAL
        NOT_MATCHED
        NOT_APPLICABLE

    The experiment summarizer must therefore count NOT_MATCHED
    as a mismatch instead of only looking for the legacy
    MISMATCH label.
    """

    records = [
        make_intelligence_record("MATCHED"),
        make_intelligence_record("PARTIAL"),
        make_intelligence_record("NOT_MATCHED"),
        make_intelligence_record("NOT_MATCHED"),
        make_intelligence_record("UNKNOWN"),
    ]

    summary = _summarize_decision_intelligence(records)

    assert summary["decision_intelligence_count"] == 5.0
    assert summary["matched_effect_count"] == 1.0
    assert summary["partial_effect_count"] == 1.0
    assert summary["mismatch_effect_count"] == 2.0
    assert summary["unknown_effect_count"] == 1.0
    assert summary["effect_match_rate"] == 0.2
    assert summary["mean_alternative_count"] == 2.0
    assert summary["mean_score_margin"] == 2.0


def test_decision_intelligence_summary_supports_legacy_mismatch_label():
    """
    Preserve compatibility with any older stored reports that may
    still contain the legacy MISMATCH effect label.
    """

    records = [
        make_intelligence_record("MATCHED"),
        make_intelligence_record("MISMATCH"),
        make_intelligence_record("NOT_MATCHED"),
    ]

    summary = _summarize_decision_intelligence(records)

    assert summary["decision_intelligence_count"] == 3.0
    assert summary["matched_effect_count"] == 1.0
    assert summary["partial_effect_count"] == 0.0
    assert summary["mismatch_effect_count"] == 2.0
    assert summary["unknown_effect_count"] == 0.0
    assert summary["effect_match_rate"] == 1 / 3


def test_decision_intelligence_summary_returns_zero_metrics_without_reports():
    records = [
        SimpleNamespace(outcome={}),
        SimpleNamespace(outcome={}),
    ]

    summary = _summarize_decision_intelligence(records)

    assert summary == {
        "decision_intelligence_count": 0.0,
        "matched_effect_count": 0.0,
        "partial_effect_count": 0.0,
        "mismatch_effect_count": 0.0,
        "unknown_effect_count": 0.0,
        "effect_match_rate": 0.0,
        "mean_alternative_count": 0.0,
        "mean_score_margin": 0.0,
    }
