#!/usr/bin/env python3

from collections import Counter

from nexus.agents.nexus.runner import NexusAgentRunner


def main() -> None:
    runner = NexusAgentRunner(
        episode_steps=720,
        seed=101,
        debug=False,
    )

    result = runner.run()

    records = result["memory"]
    decision_intelligence = result["decision_intelligence"]
    experiences = runner.experience_store.all()

    effect_counts = Counter(
        experience.effect_match or "UNKNOWN"
        for experience in experiences
    )

    interpretations = [
        record.metadata["evidence_interpretation"]
        for record in records
        if isinstance(
            record.metadata.get("evidence_interpretation"),
            dict,
        )
    ]

    interpretation_patterns = Counter(
        item.get("evidence_pattern", "UNKNOWN")
        for item in interpretations
    )

    successful_harvests = [
        record
        for record in records
        if record.decision_name == "HARVEST_CROP"
        and isinstance(record.outcome, dict)
        and record.outcome.get(
            "harvested_units_delta",
            0,
        ) > 0
    ]

    verified_harvests = [
        record
        for record in records
        if record.decision_name == "HARVEST_CROP"
        and isinstance(record.outcome, dict)
        and record.outcome.get(
            "target_crop_removed"
        ) is True
    ]

    matched_harvests = [
        record
        for record in records
        if record.decision_name == "HARVEST_CROP"
        and isinstance(record.outcome, dict)
        and record.outcome.get(
            "evaluation",
            {},
        ).get("effect_match") == "MATCHED"
    ]

    records_with_retrieval = [
        record
        for record in records
        if record.metadata.get(
            "retrieval_summary",
            {},
        ).get("retrieved_count", 0) > 0
    ]

    total_harvested_units = sum(
        record.outcome.get(
            "harvested_units_delta",
            0,
        )
        for record in successful_harvests
    )

    # Select a representative decision trace for the demo.
    demo_decision = next(
        (
            item
            for item in decision_intelligence
            if item.get("decision_name") == "PLANT_WHEAT"
        ),
        decision_intelligence[0],
    )

    print("=" * 78)
    print("NEXUS — INTELLIGENT FARM DECISION & AUTONOMOUS MANAGEMENT SYSTEM")
    print("=" * 78)

    print("\nMVP DEMONSTRATION")
    print("-" * 78)
    print("Environment:           Kaggriculture")
    print("Episode steps:         720")
    print("Seed:                  101")
    print("Decision records:      ", len(records))
    print("Experiences recorded:  ", len(experiences))

    print("\nLIVE DECISION TRACE")
    print("-" * 78)
    print(f"Step:                  {demo_decision['step']}")
    print(f"Decision:              {demo_decision['decision_name']}")
    print(f"Action:                {demo_decision['decision_action']}")
    print(f"Score:                 {demo_decision['decision_score']}")
    print(f"Confidence:            {demo_decision['confidence']}")
    print(f"Reason:                {demo_decision['decision_reason']}")
    print(f"Runner-up:             {demo_decision['runner_up']}")
    print(f"Runner-up score:       {demo_decision['runner_up_score']}")
    print(f"Expected effect:       {demo_decision['expected_effect']}")
    print(f"Observed effect:       {demo_decision['observed_effect']}")
    print(f"Effect verification:   {demo_decision['effect_match']}")

    evidence = demo_decision["evidence"]

    print("\nSTATE TRANSITION")
    print("-" * 78)
    print(
        f"Money:                 "
        f"{evidence['money_before']:.0f} → "
        f"{evidence['money_after']:.0f}"
    )
    print(
        f"Wheat seed:            "
        f"{evidence['seeds_before']['WHEAT']} → "
        f"{evidence['seeds_after']['WHEAT']}"
    )
    print(
        f"Crop count:            "
        f"{evidence['crop_count_before']} → "
        f"{evidence['crop_count_after']}"
    )
    print(
        f"Target crop:           "
        f"{evidence['target_crop_before']} → "
        f"{evidence['target_crop_after']['crop']}"
    )

    print("\nEXPERIENCE QUALITY")
    print("-" * 78)
    for label in (
        "MATCHED",
        "PARTIAL",
        "NOT_MATCHED",
        "UNKNOWN",
    ):
        print(
            f"{label:<20}: "
            f"{effect_counts.get(label, 0)}"
        )

    print("\nAUTONOMOUS HARVEST")
    print("-" * 78)
    print(
        f"Successful harvests:   "
        f"{len(successful_harvests)}"
    )
    print(
        f"Verified targets:      "
        f"{len(verified_harvests)}"
    )
    print(
        f"Matched harvests:      "
        f"{len(matched_harvests)}"
    )
    print(
        f"Harvested units:       "
        f"{total_harvested_units}"
    )

    print("\nHISTORICAL INTELLIGENCE")
    print("-" * 78)
    print(
        f"Records interpreted:   "
        f"{len(interpretations)}"
    )
    print(
        f"Records with retrieval: "
        f"{len(records_with_retrieval)}"
    )
    print(
        f"Patterns:              "
        f"{dict(interpretation_patterns)}"
    )

    print("\nMVP VALIDATION")
    print("-" * 78)

    checks = {
        "Experiences match decisions":
            len(experiences) == len(records),

        "Harvest targets verified":
            len(successful_harvests)
            == len(verified_harvests),

        "Harvest outcomes matched":
            len(successful_harvests)
            == len(matched_harvests),

        "Historical retrieval active":
            len(records_with_retrieval) > 0,

        "Evidence interpretation active":
            len(interpretations) > 0,
    }

    for label, passed in checks.items():
        print(
            f"{'PASS' if passed else 'FAIL':<6} "
            f"{label}"
        )

    print("\nOVERALL")
    print("-" * 78)
    print(
        "PASS"
        if all(checks.values())
        else "CHECK"
    )


if __name__ == "__main__":
    main()
