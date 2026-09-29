from collections import Counter, defaultdict

from nexus.agents.nexus.runner import NexusAgentRunner


EPISODE_STEPS = 720
DIAGNOSTIC_SEEDS = (101, 202)


def money_from_observation(observation, player_id="player_0"):
    farm = observation["farms"][player_id]
    return float(farm["money"])


def shed_from_observation(observation):
    return observation["private"]["shed"]


def _farmer_action(record):
    if isinstance(record, dict):
        executed = record.get("executed_action") or {}
    else:
        executed = getattr(record, "executed_action", None) or {}

    if isinstance(executed, dict):
        action = executed.get("farmer", [])

        if isinstance(action, list):
            return action

        if action is None:
            return []

        return [action]

    return []


def _market_actions(record):
    if isinstance(record, dict):
        executed = record.get("executed_action") or {}
    else:
        executed = getattr(record, "executed_action", None) or {}

    if isinstance(executed, dict):
        actions = executed.get("market", [])

        if isinstance(actions, list):
            return actions

        if actions is None:
            return []

        return [actions]

    return []


def _action_name(action):
    if not action:
        return None

    if isinstance(action, (list, tuple)):
        if not action:
            return None

        return str(action[0]).upper()

    return str(action).upper()


def _is_movement_action(action):
    return _action_name(action) in {
        "NORTH",
        "SOUTH",
        "EAST",
        "WEST",
    }


def _is_terminal_farmer_action(action):
    return _action_name(action) in {
        "BUY_SEED",
        "PLANT",
        "WATER",
        "HARVEST",
        "DIG",
    }


def _is_terminal_market_action(action):
    return _action_name(action) in {
        "BUY",
        "SELL",
        "BUY_SEED",
        "BUY_PRODUCT",
        "BUY_ANIMAL",
        "HIRE",
        "BUY_LAND",
    }


def _is_strategic_spatial_intent(decision_name):
    if not decision_name:
        return False

    decision_name = str(decision_name).upper()

    return (
        decision_name == "HARVEST_CROP"
        or decision_name == "CLEAR_WEED"
        or decision_name.startswith("PLANT_")
    )


def _is_day_boundary(report):
    evidence = report.get("evidence", {})

    if not isinstance(evidence, dict):
        return False

    hour_before = evidence.get("hour_before")
    hour_after = evidence.get("hour_after")

    return hour_before == 23 and hour_after == 0


def _record_outcome(record):
    """Return the raw runner outcome stored on a memory record."""

    if record is None:
        return {}

    if isinstance(record, dict):
        outcome = record.get("outcome", {})
    else:
        outcome = getattr(record, "outcome", {})

    return outcome if isinstance(outcome, dict) else {}


def _memory_record_for_step(memory, step):
    """Find the memory record whose stored decision step matches *step*."""

    for record in memory:
        if isinstance(record, dict):
            record_step = record.get("step")
        else:
            record_step = getattr(record, "step", None)

        if record_step == step:
            return record

    return None


def _report_evidence(report, record=None):
    """
    Return the diagnostic evidence view for a decision.

    The DecisionIntelligence report contains evaluator evidence under
    ``report["evidence"]``. The runner's exact target-level transition
    evidence is stored separately on the raw memory outcome. Merge both
    sources, with raw runner outcome fields taking precedence.
    """

    evidence = report.get("evidence", {})

    if not isinstance(evidence, dict):
        evidence = {}
    else:
        evidence = dict(evidence)

    outcome = _record_outcome(record)

    if outcome:
        evidence.update(outcome)

    return evidence


def _execution_intent_metadata(record):
    if record is None:
        return {}

    if isinstance(record, dict):
        metadata = record.get("metadata")
    else:
        metadata = getattr(record, "metadata", None)

    if not isinstance(metadata, dict):
        return {}

    intent = metadata.get("execution_intent")

    if isinstance(intent, dict):
        return intent

    return {}


def _execution_target(record):
    intent = _execution_intent_metadata(record)

    target = intent.get("target")

    if not isinstance(target, dict):
        metadata = getattr(record, "metadata", None)

        if isinstance(metadata, dict):
            target = metadata.get("planning_target")

    if not isinstance(target, dict):
        return None

    if "x" not in target or "y" not in target:
        return None

    try:
        return {
            "x": int(target["x"]),
            "y": int(target["y"]),
        }
    except (TypeError, ValueError):
        return None


def _runner_target_evidence(report, record=None):
    """
    Read exact target-level evidence from the runner's raw memory outcome.

    NexusAgentRunner stores these fields at the TOP LEVEL of
    ``memory_record["outcome"]`` rather than inside the serialized
    DecisionIntelligence ``report["evidence"]`` dictionary.

    Report evidence remains a compatibility fallback for older records.
    """

    outcome = _record_outcome(record)
    report_evidence = report.get("evidence", {})

    if not isinstance(report_evidence, dict):
        report_evidence = {}

    def value(key):
        if key in outcome:
            return outcome.get(key)
        return report_evidence.get(key)

    return {
        "execution_target": value("execution_target"),
        "target_crop_before": value("target_crop_before"),
        "target_crop_after": value("target_crop_after"),
        "target_yield_before": value("target_yield_before"),
        "target_yield_after": value("target_yield_after"),
        "target_crop_removed": value("target_crop_removed"),
        "target_yield_reduced": value("target_yield_reduced"),
        "harvested_product": value("harvested_product"),
        "harvested_units_before": value("harvested_units_before"),
        "harvested_units_after": value("harvested_units_after"),
        "harvested_units_delta": value("harvested_units_delta"),
    }


def _effective_execution_target(report, record):
    """
    Prefer target evidence captured by the runner.

    Fall back to ExecutionIntent metadata for records produced by
    older runner output.
    """

    evidence = _runner_target_evidence(report, record)

    target = evidence.get("execution_target")

    if isinstance(target, dict):
        if "x" in target and "y" in target:
            try:
                return {
                    "x": int(target["x"]),
                    "y": int(target["y"]),
                }
            except (TypeError, ValueError):
                pass

    return _execution_target(record)


def _crop_rows(report, field):
    evidence = _report_evidence(report)

    crops = evidence.get(field, [])

    if not isinstance(crops, list):
        return []

    return [
        crop
        for crop in crops
        if isinstance(crop, dict)
    ]


def _crop_at_target(report, field, target):
    if target is None:
        return None

    try:
        target_x = int(target["x"])
        target_y = int(target["y"])
    except (TypeError, ValueError, KeyError):
        return None

    for crop in _crop_rows(report, field):
        try:
            crop_x = int(crop.get("col"))
            crop_y = int(crop.get("row"))
        except (TypeError, ValueError):
            continue

        if crop_x == target_x and crop_y == target_y:
            return crop

    return None


def _shed_deltas(report, record=None):
    evidence = _report_evidence(report, record)

    before = evidence.get("shed_before", {})
    after = evidence.get("shed_after", {})

    if not isinstance(before, dict):
        before = {}

    if not isinstance(after, dict):
        after = {}

    item_names = set(before) | set(after)
    deltas = {}

    for item in item_names:
        try:
            delta = int(after.get(item, 0)) - int(
                before.get(item, 0)
            )
        except (TypeError, ValueError):
            continue

        if delta != 0:
            deltas[item] = delta

    return deltas


def _seed_deltas(report):
    evidence = _report_evidence(report)

    before = evidence.get("seeds_before", {})
    after = evidence.get("seeds_after", {})

    if not isinstance(before, dict):
        before = {}

    if not isinstance(after, dict):
        after = {}

    item_names = set(before) | set(after)
    deltas = {}

    for item in item_names:
        try:
            delta = int(after.get(item, 0)) - int(
                before.get(item, 0)
            )
        except (TypeError, ValueError):
            continue

        if delta != 0:
            deltas[item] = delta

    return deltas


def _crop_count_delta(report):
    evidence = _report_evidence(report)

    value = evidence.get("crop_count_delta")

    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass

    before = evidence.get("crop_count_before")
    after = evidence.get("crop_count_after")

    try:
        if before is not None and after is not None:
            return int(after) - int(before)
    except (TypeError, ValueError):
        pass

    return 0


def _weed_count_delta(report):
    evidence = _report_evidence(report)

    value = evidence.get("weed_count_delta")

    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass

    before = evidence.get("weed_count_before")
    after = evidence.get("weed_count_after")

    try:
        if before is not None and after is not None:
            return int(after) - int(before)
    except (TypeError, ValueError):
        pass

    return 0


def _money_delta(report):
    evidence = _report_evidence(report)

    value = evidence.get("money_delta")

    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            pass

    before = evidence.get("money_before")
    after = evidence.get("money_after")

    try:
        if before is not None and after is not None:
            return float(after) - float(before)
    except (TypeError, ValueError):
        pass

    return 0.0


def _harvest_target_verification(report, record):
    """
    Verify HARVEST_CROP using the strongest target-level evidence.

    Kaggriculture places harvested product into the farmer's
    immediate inventory first. Transfer to the farm shed occurs
    later at end-of-day.

    VERIFIED requires:

        target crop existed before
        AND
        target crop was removed OR target yield decreased
        AND
        harvested product entered immediate inventory

    Shed gain remains a backward-compatible fallback for older
    outcome records.
    """

    runner_evidence = _runner_target_evidence(report, record)

    target = runner_evidence.get("execution_target")

    if not isinstance(target, dict):
        target = _effective_execution_target(report, record)

    target_before = runner_evidence.get(
        "target_crop_before"
    )

    target_crop_removed = runner_evidence.get(
        "target_crop_removed"
    )

    target_yield_reduced = runner_evidence.get(
        "target_yield_reduced"
    )

    harvested_units_delta = runner_evidence.get(
        "harvested_units_delta",
        0,
    )

    has_harvest_gain = (
        isinstance(
            harvested_units_delta,
            (int, float),
        )
        and harvested_units_delta > 0
    )

    shed_deltas = _shed_deltas(report, record)

    has_shed_gain = any(
        delta > 0
        for delta in shed_deltas.values()
    )

    harvest_state_changed = (
        target_crop_removed is True
        or target_yield_reduced is True
    )

    # -------------------------------------------------------------
    # PRIMARY: immediate farmer inventory evidence
    # -------------------------------------------------------------

    if (
        target is not None
        and target_before is not None
        and harvest_state_changed
        and has_harvest_gain
    ):
        return "VERIFIED"

    # -------------------------------------------------------------
    # FALLBACK: older records where shed evidence was captured
    # -------------------------------------------------------------

    if (
        target is not None
        and target_before is not None
        and harvest_state_changed
        and has_shed_gain
    ):
        return "VERIFIED"

    if (
        target_before is not None
        and harvest_state_changed
        and (
            has_harvest_gain
            or has_shed_gain
        )
    ):
        return "VERIFIED"

    # -------------------------------------------------------------
    # Evidence exists but harvest completion is not fully proven
    # -------------------------------------------------------------

    if (
        harvest_state_changed
        or has_harvest_gain
        or has_shed_gain
    ):
        return "UNCONFIRMED"

    return "UNCONFIRMED"

def _verify_terminal_spatial_action(report, record):
    """
    Verify terminal spatial actions.

    Returns:

        VERIFIED
            Target-level evidence confirms the intended action.

        VERIFIED_EFFECT
            Aggregate state evidence confirms the effect but not
            target-specific causality.

        UNCONFIRMED
            A terminal action occurred but available evidence does
            not prove the intended transition.

        NOT_APPLICABLE
            The record is not a spatial terminal action.
    """

    decision = str(
        report.get("decision_name") or ""
    ).upper()

    farmer_action = _farmer_action(record)
    farmer_name = _action_name(farmer_action)

    if farmer_name not in {
        "HARVEST",
        "PLANT",
        "DIG",
    }:
        return "NOT_APPLICABLE"

    # -------------------------------------------------------------
    # PLANTING
    # -------------------------------------------------------------

    if decision.startswith("PLANT_"):
        intended_crop = decision.replace(
            "PLANT_",
            "",
            1,
        )

        target = _effective_execution_target(
            report,
            record,
        )

        target_evidence = _runner_target_evidence(
            report,
            record,
        )

        target_after = target_evidence.get(
            "target_crop_after"
        )

        if (
            target_after is not None
            and str(
                target_after.get("crop", "")
            ).upper()
            == intended_crop
        ):
            return "VERIFIED"

        if target is not None:
            target_after = _crop_at_target(
                report,
                "crops_after",
                target,
            )

            if (
                target_after is not None
                and str(
                    target_after.get("crop", "")
                ).upper()
                == intended_crop
            ):
                return "VERIFIED"

        crop_count_delta = _crop_count_delta(report)
        seed_deltas = _seed_deltas(report)

        seed_delta = seed_deltas.get(
            intended_crop,
            0,
        )

        if crop_count_delta > 0 and seed_delta < 0:
            return "VERIFIED"

        return "UNCONFIRMED"

    # -------------------------------------------------------------
    # HARVESTING
    # -------------------------------------------------------------

    if decision == "HARVEST_CROP":
        return _harvest_target_verification(
            report,
            record,
        )

    # -------------------------------------------------------------
    # WEED CLEARING
    # -------------------------------------------------------------

    if decision in {
        "CLEAR_WEED",
        "DIG_WEED",
    }:
        weed_delta = _weed_count_delta(report)

        if weed_delta < 0:
            return "VERIFIED_EFFECT"

        return "UNCONFIRMED"

    return "UNCONFIRMED"


def _terminal_verification_label(report, record):
    verification = _verify_terminal_spatial_action(
        report,
        record,
    )

    if verification == "VERIFIED":
        return "TARGET_VERIFIED"

    if verification == "VERIFIED_EFFECT":
        return "EFFECT_VERIFIED"

    if verification == "UNCONFIRMED":
        return "UNCONFIRMED"

    return "NOT_APPLICABLE"


def _classify_execution(report, record):
    decision = str(
        report.get("decision_name") or ""
    )

    farmer_action = _farmer_action(record)
    market_actions = _market_actions(record)

    farmer_name = _action_name(farmer_action)

    market_names = [
        _action_name(action)
        for action in market_actions
        if action
    ]

    if _is_day_boundary(report):
        return "ENVIRONMENT_TRANSITION"

    if _is_strategic_spatial_intent(decision):
        if _is_movement_action(farmer_action):
            return "NAVIGATION"

        if farmer_name in {
            "HARVEST",
            "PLANT",
            "DIG",
        }:
            return "TERMINAL_ACTION"

    if _is_terminal_farmer_action(farmer_action):
        return "TERMINAL_ACTION"

    if any(
        name in {
            "BUY",
            "SELL",
            "BUY_SEED",
            "BUY_PRODUCT",
            "BUY_ANIMAL",
            "HIRE",
            "BUY_LAND",
        }
        for name in market_names
    ):
        return "TERMINAL_ACTION"

    if _is_movement_action(farmer_action):
        return "NAVIGATION"

    return "NO_TERMINAL_ACTION"


def _classify_outcome(report, record, execution_stage):
    effect = report.get("effect_match")

    if execution_stage == "ENVIRONMENT_TRANSITION":
        return "ENVIRONMENT_TRANSITION"

    if execution_stage == "NAVIGATION":
        return "IN_PROGRESS"

    if execution_stage == "TERMINAL_ACTION":
        verification = _verify_terminal_spatial_action(
            report,
            record,
        )

        if verification in {
            "VERIFIED",
            "VERIFIED_EFFECT",
        }:
            return "COMPLETED"

        if effect == "MATCHED":
            return "COMPLETED"

        if effect == "NOT_MATCHED":
            return "FAILED"

        if effect == "PARTIAL":
            return "TERMINAL_UNCONFIRMED"

        return "TERMINAL_UNCONFIRMED"

    if effect == "MATCHED":
        return "COMPLETED"

    if effect == "NOT_MATCHED":
        return "FAILED"

    if effect == "PARTIAL":
        return "PARTIAL"

    return "UNKNOWN"


def _diagnostic_records(intelligence, memory):
    records = []

    for report in intelligence:
        record = _memory_record_for_step(
            memory,
            report.get("step"),
        )

        if record is not None:
            execution_stage = _classify_execution(
                report,
                record,
            )

            verification = _terminal_verification_label(
                report,
                record,
            )

        else:
            execution_stage = "NO_MEMORY_RECORD"
            verification = "NOT_APPLICABLE"

        diagnostic_status = _classify_outcome(
            report,
            record,
            execution_stage,
        )

        records.append(
            {
                "report": report,
                "record": record,
                "execution_stage": execution_stage,
                "diagnostic_status": diagnostic_status,
                "terminal_verification": verification,
            }
        )

    return records


def _print_counter(title, counter, total=None, width=30):
    print(f"\n{title}")
    print("-" * 100)

    for name, count in counter.most_common():
        if total:
            percentage = count / total * 100

            print(
                f"{str(name):{width}} "
                f"{count:4} "
                f"({percentage:6.2f}%)"
            )
        else:
            print(
                f"{str(name):{width}} "
                f"{count:4}"
            )


def _print_action(action):
    return str(action) if action else "[]"


def _print_harvest_evidence(item):
    """
    Print target-level harvest evidence for diagnostic inspection.
    """

    report = item["report"]
    record = item["record"]

    evidence = _runner_target_evidence(report, record)
    target = evidence.get("execution_target")

    if target is None:
        target = _execution_target(record)

    if target is None:
        print("  target:          unavailable")
        return

    print(
        f"  target:          "
        f"(x={target.get('x')}, y={target.get('y')})"
    )

    print(
        f"  crop_before:     "
        f"{evidence.get('target_crop_before')}"
    )

    print(
        f"  crop_after:      "
        f"{evidence.get('target_crop_after')}"
    )

    print(
        f"  yield_before:    "
        f"{evidence.get('target_yield_before')}"
    )

    print(
        f"  yield_after:     "
        f"{evidence.get('target_yield_after')}"
    )

    print(
        f"  crop_removed:    "
        f"{evidence.get('target_crop_removed')}"
    )

    print(
        f"  yield_reduced:   "
        f"{evidence.get('target_yield_reduced')}"
    )

    print(
        f"  shed_delta:      "
        f"{_shed_deltas(report, record)}"
    )


def summarize_seed(seed: int):
    runner = NexusAgentRunner(
        episode_steps=EPISODE_STEPS,
        seed=seed,
        debug=False,
    )

    result = runner.run()

    intelligence = result["decision_intelligence"]
    memory = result["memory"]
    evaluation = result["evaluation"]

    diagnostic_records = _diagnostic_records(
        intelligence,
        memory,
    )

    print("\n" + "=" * 100)
    print(
        f"SEED {seed} — "
        f"NEXUS MVP DECISION QUALITY DIAGNOSTIC"
    )
    print("=" * 100)

    # ------------------------------------------------------------------
    # 1. BASIC EPISODE INFORMATION
    # ------------------------------------------------------------------

    print("\nEPISODE")
    print("-" * 100)

    print(
        f"Steps requested:       "
        f"{EPISODE_STEPS}"
    )

    print(
        f"Decisions recorded:    "
        f"{len(memory)}"
    )

    print(
        f"Intelligence reports:  "
        f"{len(intelligence)}"
    )

    print(
        f"Experiences recorded:  "
        f"{result['experience_count']}"
    )

    # ------------------------------------------------------------------
    # 2. RAW EVALUATOR EFFECT QUALITY
    # ------------------------------------------------------------------

    effect_counts = Counter(
        report.get("effect_match")
        for report in intelligence
    )

    total = len(intelligence)

    print("\nRAW EVALUATOR EFFECT QUALITY")
    print("-" * 100)

    for key in [
        "MATCHED",
        "PARTIAL",
        "NOT_MATCHED",
    ]:
        count = effect_counts.get(key, 0)

        percentage = (
            count / total * 100
            if total
            else 0
        )

        print(
            f"{key:20} "
            f"{count:4} "
            f"({percentage:6.2f}%)"
        )

    print(
        "\nNOTE: RAW PARTIAL is not automatically a failure."
    )

    print(
        "A PARTIAL result may represent navigation, "
        "partial state evidence, or an ongoing strategic intent."
    )

    # ------------------------------------------------------------------
    # 3. MVP EXECUTION STAGES
    # ------------------------------------------------------------------

    execution_counts = Counter(
        item["execution_stage"]
        for item in diagnostic_records
    )

    _print_counter(
        "MVP EXECUTION STAGES",
        execution_counts,
        total,
        width=32,
    )

    print(
        "\nInterpretation:"
        "\n  NAVIGATION              = strategic intent is being executed spatially"
        "\n  TERMINAL_ACTION         = physical/market action was actually attempted"
        "\n  ENVIRONMENT_TRANSITION  = day rollover/environment boundary"
        "\n  NO_TERMINAL_ACTION      = no terminal operation detected in this record"
    )

    # ------------------------------------------------------------------
    # 4. DIAGNOSTIC OUTCOME STATUS
    # ------------------------------------------------------------------

    status_counts = Counter(
        item["diagnostic_status"]
        for item in diagnostic_records
    )

    _print_counter(
        "DIAGNOSTIC OUTCOME STATUS",
        status_counts,
        total,
        width=32,
    )

    print(
        "\nThis section is the primary MVP interpretation layer."
    )

    # ------------------------------------------------------------------
    # 5. TERMINAL VERIFICATION QUALITY
    # ------------------------------------------------------------------

    verification_counts = Counter(
        item["terminal_verification"]
        for item in diagnostic_records
        if item["execution_stage"] == "TERMINAL_ACTION"
    )

    _print_counter(
        "TERMINAL VERIFICATION QUALITY",
        verification_counts,
        sum(verification_counts.values()),
        width=32,
    )

    print(
        "\nInterpretation:"
        "\n  TARGET_VERIFIED         = terminal action is confirmed against its target"
        "\n  EFFECT_VERIFIED         = terminal effect is confirmed, but target causality is not proven"
        "\n  UNCONFIRMED             = terminal action occurred but evidence is insufficient"
    )

    # ------------------------------------------------------------------
    # 6. DECISION DISTRIBUTION
    # ------------------------------------------------------------------

    decision_counts = Counter(
        report.get("decision_name")
        for report in intelligence
    )

    _print_counter(
        "STRATEGIC DECISION DISTRIBUTION",
        decision_counts,
        total,
        width=32,
    )

    # ------------------------------------------------------------------
    # 7. EXECUTED FARMER ACTION DISTRIBUTION
    # ------------------------------------------------------------------

    action_counts = Counter()

    for record in memory:
        farmer_action = _farmer_action(record)

        if farmer_action:
            action_counts[
                _print_action(farmer_action)
            ] += 1

    _print_counter(
        "EXECUTED FARMER ACTIONS",
        action_counts,
        total,
        width=38,
    )

    # ------------------------------------------------------------------
    # 8. MARKET ACTION DISTRIBUTION
    # ------------------------------------------------------------------

    market_action_counts = Counter()

    for record in memory:
        for action in _market_actions(record):
            if action:
                market_action_counts[
                    _print_action(action)
                ] += 1

    _print_counter(
        "EXECUTED MARKET ACTIONS",
        market_action_counts,
        total,
        width=38,
    )

    # ------------------------------------------------------------------
    # 9. DECISION → EXECUTION MATRIX
    # ------------------------------------------------------------------

    decision_execution = defaultdict(Counter)

    for item in diagnostic_records:
        decision = item["report"].get("decision_name")
        stage = item["execution_stage"]

        decision_execution[decision][stage] += 1

    print("\nDECISION → EXECUTION MATRIX")
    print("-" * 100)

    for decision, stages in sorted(
        decision_execution.items()
    ):
        decision_total = sum(stages.values())

        navigation_count = stages.get(
            "NAVIGATION",
            0,
        )

        terminal_count = stages.get(
            "TERMINAL_ACTION",
            0,
        )

        transition_count = stages.get(
            "ENVIRONMENT_TRANSITION",
            0,
        )

        other_count = sum(
            count
            for stage, count in stages.items()
            if stage not in {
                "NAVIGATION",
                "TERMINAL_ACTION",
                "ENVIRONMENT_TRANSITION",
            }
        )

        print(
            f"{str(decision):30} "
            f"total={decision_total:3} "
            f"navigation={navigation_count:3} "
            f"terminal={terminal_count:3} "
            f"transition={transition_count:3} "
            f"other={other_count:3}"
        )

    # ------------------------------------------------------------------
    # 10. DECISION → OUTCOME MATRIX
    # ------------------------------------------------------------------

    decision_status = defaultdict(Counter)

    for item in diagnostic_records:
        decision = item["report"].get("decision_name")
        status = item["diagnostic_status"]

        decision_status[decision][status] += 1

    print("\nDECISION → DIAGNOSTIC OUTCOME")
    print("-" * 100)

    for decision, statuses in sorted(
        decision_status.items()
    ):
        decision_total = sum(statuses.values())

        completed = statuses.get(
            "COMPLETED",
            0,
        )

        in_progress = statuses.get(
            "IN_PROGRESS",
            0,
        )

        failed = statuses.get(
            "FAILED",
            0,
        )

        transition = statuses.get(
            "ENVIRONMENT_TRANSITION",
            0,
        )

        terminal_unconfirmed = statuses.get(
            "TERMINAL_UNCONFIRMED",
            0,
        )

        completion_pct = (
            completed / decision_total * 100
            if decision_total
            else 0
        )

        print(
            f"{str(decision):30} "
            f"total={decision_total:3} "
            f"completed={completed:3} "
            f"in_progress={in_progress:3} "
            f"terminal_unconfirmed={terminal_unconfirmed:3} "
            f"failed={failed:3} "
            f"transition={transition:3} "
            f"completion={completion_pct:6.2f}%"
        )

    # ------------------------------------------------------------------
    # 11. TERMINAL ACTION ANALYSIS
    # ------------------------------------------------------------------

    terminal_records = [
        item
        for item in diagnostic_records
        if item["execution_stage"] == "TERMINAL_ACTION"
    ]

    terminal_status_counts = Counter(
        item["diagnostic_status"]
        for item in terminal_records
    )

    print("\nTERMINAL ACTION ANALYSIS")
    print("-" * 100)

    print(
        f"Terminal action records: "
        f"{len(terminal_records)}"
    )

    for status, count in terminal_status_counts.most_common():
        percentage = (
            count / len(terminal_records) * 100
            if terminal_records
            else 0
        )

        print(
            f"  {status:28} "
            f"{count:4} "
            f"({percentage:6.2f}%)"
        )

    # ------------------------------------------------------------------
    # 12. PERSISTENT SPATIAL INTENT ANALYSIS
    # ------------------------------------------------------------------

    spatial_records = [
        item
        for item in diagnostic_records
        if _is_strategic_spatial_intent(
            item["report"].get("decision_name")
        )
    ]

    spatial_stages = Counter(
        item["execution_stage"]
        for item in spatial_records
    )

    spatial_status = Counter(
        item["diagnostic_status"]
        for item in spatial_records
    )

    print("\nPERSISTENT SPATIAL INTENT ANALYSIS")
    print("-" * 100)

    print(
        f"Spatial strategic records: "
        f"{len(spatial_records)}"
    )

    print(
        f"  Navigation/progress: "
        f"{spatial_stages.get('NAVIGATION', 0)}"
    )

    print(
        f"  Terminal execution: "
        f"{spatial_stages.get('TERMINAL_ACTION', 0)}"
    )

    print(
        f"  Completed: "
        f"{spatial_status.get('COMPLETED', 0)}"
    )

    print(
        f"  In progress: "
        f"{spatial_status.get('IN_PROGRESS', 0)}"
    )

    print(
        f"  Failed: "
        f"{spatial_status.get('FAILED', 0)}"
    )

    # ------------------------------------------------------------------
    # 13. SPATIAL TERMINAL VERIFICATION
    # ------------------------------------------------------------------

    spatial_terminal_records = [
        item
        for item in spatial_records
        if item["execution_stage"] == "TERMINAL_ACTION"
    ]

    spatial_verification = Counter(
        item["terminal_verification"]
        for item in spatial_terminal_records
    )

    print("\nSPATIAL TERMINAL VERIFICATION")
    print("-" * 100)

    print(
        f"Spatial terminal records: "
        f"{len(spatial_terminal_records)}"
    )

    print(
        f"  Target verified: "
        f"{spatial_verification.get('TARGET_VERIFIED', 0)}"
    )

    print(
        f"  Effect verified: "
        f"{spatial_verification.get('EFFECT_VERIFIED', 0)}"
    )

    print(
        f"  Unconfirmed: "
        f"{spatial_verification.get('UNCONFIRMED', 0)}"
    )

    print(
        "\nTarget verification now prefers target-level evidence "
        "captured directly by NexusAgentRunner."
    )

    print(
        "Weed clearing still uses aggregate weed-count evidence, "
        "so DIG can be effect-verified without claiming "
        "target-level causal verification."
    )

    # ------------------------------------------------------------------
    # 14. PLANTING ANALYSIS
    # ------------------------------------------------------------------

    planting_records = [
        item
        for item in diagnostic_records
        if str(
            item["report"].get("decision_name", "")
        ).startswith("PLANT_")
    ]

    planting_stages = Counter(
        item["execution_stage"]
        for item in planting_records
    )

    planting_status = Counter(
        item["diagnostic_status"]
        for item in planting_records
    )

    planting_verification = Counter(
        item["terminal_verification"]
        for item in planting_records
        if item["execution_stage"] == "TERMINAL_ACTION"
    )

    print("\nPLANTING ANALYSIS")
    print("-" * 100)

    print(
        f"Planting decisions: "
        f"{len(planting_records)}"
    )

    print(
        f"  Navigation: "
        f"{planting_stages.get('NAVIGATION', 0)}"
    )

    print(
        f"  Terminal PLANT actions: "
        f"{planting_stages.get('TERMINAL_ACTION', 0)}"
    )

    print(
        f"  Completed: "
        f"{planting_status.get('COMPLETED', 0)}"
    )

    print(
        f"  In progress: "
        f"{planting_status.get('IN_PROGRESS', 0)}"
    )

    print(
        f"  Failed: "
        f"{planting_status.get('FAILED', 0)}"
    )

    print(
        f"  Terminal unconfirmed: "
        f"{planting_status.get('TERMINAL_UNCONFIRMED', 0)}"
    )

    print(
        f"  Target verified: "
        f"{planting_verification.get('TARGET_VERIFIED', 0)}"
    )

    # ------------------------------------------------------------------
    # 15. CROP-SPECIFIC PLANTING
    # ------------------------------------------------------------------

    crop_planting = defaultdict(Counter)

    for item in planting_records:
        decision = str(
            item["report"].get("decision_name", "")
        )

        crop = decision.replace(
            "PLANT_",
            "",
            1,
        )

        crop_planting[crop][
            item["diagnostic_status"]
        ] += 1

    print("\nCROP-SPECIFIC PLANTING RESULTS")
    print("-" * 100)

    for crop, statuses in sorted(
        crop_planting.items()
    ):
        total_crop = sum(statuses.values())

        print(
            f"{crop:15} "
            f"total={total_crop:3} "
            f"completed={statuses.get('COMPLETED', 0):3} "
            f"in_progress={statuses.get('IN_PROGRESS', 0):3} "
            f"terminal_unconfirmed="
            f"{statuses.get('TERMINAL_UNCONFIRMED', 0):3} "
            f"failed={statuses.get('FAILED', 0):3}"
        )

    # ------------------------------------------------------------------
    # 16. HARVEST ANALYSIS
    # ------------------------------------------------------------------

    harvest_records = [
        item
        for item in diagnostic_records
        if item["report"].get("decision_name")
        == "HARVEST_CROP"
    ]

    harvest_stages = Counter(
        item["execution_stage"]
        for item in harvest_records
    )

    harvest_status = Counter(
        item["diagnostic_status"]
        for item in harvest_records
    )

    harvest_verification = Counter(
        item["terminal_verification"]
        for item in harvest_records
        if item["execution_stage"] == "TERMINAL_ACTION"
    )

    harvest_target_evidence = Counter()

    for item in harvest_records:
        if item["execution_stage"] != "TERMINAL_ACTION":
            continue

        evidence = _runner_target_evidence(
            item["report"],
            item["record"],
        )

        target = evidence.get(
            "execution_target"
        )

        before = evidence.get(
            "target_crop_before"
        )

        removed = evidence.get(
            "target_crop_removed"
        )

        reduced = evidence.get(
            "target_yield_reduced"
        )

        if target is not None:
            harvest_target_evidence[
                "target_captured"
            ] += 1

        if before is not None:
            harvest_target_evidence[
                "crop_present_before"
            ] += 1

        if removed is True:
            harvest_target_evidence[
                "crop_removed"
            ] += 1

        if reduced is True:
            harvest_target_evidence[
                "yield_reduced"
            ] += 1

    print("\nHARVEST ANALYSIS")
    print("-" * 100)

    print(
        f"Harvest decisions: "
        f"{len(harvest_records)}"
    )

    print(
        f"  Navigation: "
        f"{harvest_stages.get('NAVIGATION', 0)}"
    )

    print(
        f"  Terminal HARVEST actions: "
        f"{harvest_stages.get('TERMINAL_ACTION', 0)}"
    )

    print(
        f"  Completed: "
        f"{harvest_status.get('COMPLETED', 0)}"
    )

    print(
        f"  In progress: "
        f"{harvest_status.get('IN_PROGRESS', 0)}"
    )

    print(
        f"  Terminal unconfirmed: "
        f"{harvest_status.get('TERMINAL_UNCONFIRMED', 0)}"
    )

    print(
        f"  Failed: "
        f"{harvest_status.get('FAILED', 0)}"
    )

    print(
        f"  Target verified: "
        f"{harvest_verification.get('TARGET_VERIFIED', 0)}"
    )

    print(
        f"  Unconfirmed terminal harvests: "
        f"{harvest_verification.get('UNCONFIRMED', 0)}"
    )

    print("\nHARVEST TARGET EVIDENCE")
    print("-" * 100)

    print(
        "  Source: NexusAgentRunner raw memory outcome "
        "(memory[*][\"outcome\"])."
    )

    print(
        f"  Target captured:       "
        f"{harvest_target_evidence.get('target_captured', 0)}"
    )

    print(
        f"  Crop present before:   "
        f"{harvest_target_evidence.get('crop_present_before', 0)}"
    )

    print(
        f"  Crop removed:          "
        f"{harvest_target_evidence.get('crop_removed', 0)}"
    )

    print(
        f"  Yield reduced:         "
        f"{harvest_target_evidence.get('yield_reduced', 0)}"
    )

    # ------------------------------------------------------------------
    # 17. WEED CLEARING ANALYSIS
    # ------------------------------------------------------------------

    weed_records = [
        item
        for item in diagnostic_records
        if item["report"].get("decision_name")
        in {
            "CLEAR_WEED",
            "DIG_WEED",
        }
    ]

    weed_stages = Counter(
        item["execution_stage"]
        for item in weed_records
    )

    weed_status = Counter(
        item["diagnostic_status"]
        for item in weed_records
    )

    weed_verification = Counter(
        item["terminal_verification"]
        for item in weed_records
        if item["execution_stage"] == "TERMINAL_ACTION"
    )

    print("\nWEED CLEARING ANALYSIS")
    print("-" * 100)

    print(
        f"Weed-clearing decisions: "
        f"{len(weed_records)}"
    )

    print(
        f"  Navigation: "
        f"{weed_stages.get('NAVIGATION', 0)}"
    )

    print(
        f"  Terminal DIG actions: "
        f"{weed_stages.get('TERMINAL_ACTION', 0)}"
    )

    print(
        f"  Completed: "
        f"{weed_status.get('COMPLETED', 0)}"
    )

    print(
        f"  In progress: "
        f"{weed_status.get('IN_PROGRESS', 0)}"
    )

    print(
        f"  Failed: "
        f"{weed_status.get('FAILED', 0)}"
    )

    print(
        f"  Effect verified: "
        f"{weed_verification.get('EFFECT_VERIFIED', 0)}"
    )

    print(
        f"  Target unconfirmed: "
        f"{weed_verification.get('UNCONFIRMED', 0)}"
    )

    # ------------------------------------------------------------------
    # 18. WATERING / ENVIRONMENT TRANSITION ANALYSIS
    # ------------------------------------------------------------------

    watering_records = [
        item
        for item in diagnostic_records
        if item["report"].get("decision_name")
        == "WATER_CROP"
    ]

    watering_status = Counter(
        item["diagnostic_status"]
        for item in watering_records
    )

    watering_transition_records = [
        item
        for item in watering_records
        if item["execution_stage"]
        == "ENVIRONMENT_TRANSITION"
    ]

    print("\nWATERING ANALYSIS")
    print("-" * 100)

    print(
        f"Watering decisions: "
        f"{len(watering_records)}"
    )

    print(
        f"  Completed: "
        f"{watering_status.get('COMPLETED', 0)}"
    )

    print(
        f"  Failed: "
        f"{watering_status.get('FAILED', 0)}"
    )

    print(
        f"  Environment transitions: "
        f"{len(watering_transition_records)}"
    )

    print(
        "\nDay-boundary watering records are reported separately "
        "because 23:00 → 00:00 resets daily state."
    )

    # ------------------------------------------------------------------
    # 19. SELLING ANALYSIS
    # ------------------------------------------------------------------

    selling_records = [
        item
        for item in diagnostic_records
        if item["report"].get("decision_name")
        == "SELL_WHEAT"
    ]

    selling_terminal = []
    selling_other = []

    for item in selling_records:
        record = item["record"]

        if record is None:
            selling_other.append(item)
            continue

        market_actions = _market_actions(record)

        has_sell = any(
            _action_name(action) == "SELL"
            for action in market_actions
        )

        if has_sell:
            selling_terminal.append(item)
        else:
            selling_other.append(item)

    selling_status = Counter(
        item["diagnostic_status"]
        for item in selling_records
    )

    print("\nSELLING ANALYSIS")
    print("-" * 100)

    print(
        f"SELL_WHEAT decisions: "
        f"{len(selling_records)}"
    )

    print(
        f"  Terminal SELL actions observed: "
        f"{len(selling_terminal)}"
    )

    print(
        f"  No terminal SELL action observed: "
        f"{len(selling_other)}"
    )

    print(
        f"  Evaluator-confirmed completed: "
        f"{selling_status.get('COMPLETED', 0)}"
    )

    print(
        "\nSELL_WHEAT is evaluated through the dedicated SELL "
        "effect-matching path in the current outcome evaluator."
    )

    # ------------------------------------------------------------------
    # 20. RAW NOT-MATCHED ANALYSIS
    # ------------------------------------------------------------------

    failed_reports = [
        report
        for report in intelligence
        if report.get("effect_match")
        == "NOT_MATCHED"
    ]

    print("\nRAW NOT-MATCHED DECISIONS")
    print("-" * 100)

    print(
        f"Total RAW NOT_MATCHED: "
        f"{len(failed_reports)}"
    )

    failed_decisions = Counter(
        report.get("decision_name")
        for report in failed_reports
    )

    for decision, count in failed_decisions.most_common():
        print(
            f"  {str(decision):30} "
            f"{count}"
        )

    # ------------------------------------------------------------------
    # 21. GENUINE DIAGNOSTIC FAILURES
    # ------------------------------------------------------------------

    genuine_failures = [
        item
        for item in diagnostic_records
        if item["diagnostic_status"] == "FAILED"
    ]

    failure_decisions = Counter(
        item["report"].get("decision_name")
        for item in genuine_failures
    )

    print("\nGENUINE DIAGNOSTIC FAILURES")
    print("-" * 100)

    print(
        f"Total genuine failures: "
        f"{len(genuine_failures)}"
    )

    if failure_decisions:
        for decision, count in failure_decisions.most_common():
            print(
                f"  {str(decision):30} "
                f"{count}"
            )
    else:
        print("  None detected.")

    # ------------------------------------------------------------------
    # 22. PASS ANALYSIS
    # ------------------------------------------------------------------

    pass_reports = [
        report
        for report in intelligence
        if report.get("decision_name")
        == "PASS"
    ]

    print("\nPASS ANALYSIS")
    print("-" * 100)

    print(
        f"PASS decisions: "
        f"{len(pass_reports)}"
    )

    if pass_reports:
        confidence_counts = Counter(
            str(report.get("confidence"))
            for report in pass_reports
        )

        print("\nPASS confidence:")

        for confidence, count in confidence_counts.most_common():
            print(
                f"  {confidence:20} "
                f"{count}"
            )

        margins = []

        for report in pass_reports:
            margin = report.get("score_margin")

            if isinstance(
                margin,
                (int, float),
            ):
                margins.append(float(margin))

        if margins:
            average_margin = sum(margins) / len(margins)
            minimum_margin = min(margins)
            maximum_margin = max(margins)

            print("\nPASS score margins:")

            print(
                f"  Average: "
                f"{average_margin:.4f}"
            )

            print(
                f"  Minimum: "
                f"{minimum_margin:.4f}"
            )

            print(
                f"  Maximum: "
                f"{maximum_margin:.4f}"
            )

    # ------------------------------------------------------------------
    # 23. ECONOMIC EVALUATION
    # ------------------------------------------------------------------

    print("\nECONOMIC EVALUATION")
    print("-" * 100)

    print(
        f"Starting money:    "
        f"{evaluation['starting_money']:.2f}"
    )

    print(
        f"Final money:       "
        f"{evaluation['final_money']:.2f}"
    )

    print(
        f"Inventory value:   "
        f"{evaluation['inventory_value']:.2f}"
    )

    print(
        f"Net profit:        "
        f"{evaluation['net_profit']:.2f}"
    )

    print(
        f"ROI:               "
        f"{evaluation['roi_percent']:.2f}%"
    )

    # ------------------------------------------------------------------
    # 24. LAST 20 DECISIONS
    # ------------------------------------------------------------------

    print("\nLAST 20 DECISIONS")
    print("-" * 100)

    for item in diagnostic_records[-20:]:
        report = item["report"]
        record = item["record"]

        farmer_action = (
            _farmer_action(record)
            if record is not None
            else []
        )

        market_actions = (
            _market_actions(record)
            if record is not None
            else []
        )

        print(
            f"step={report.get('step'):4} "
            f"decision={str(report.get('decision_name')):25} "
            f"farmer={str(farmer_action):25} "
            f"market={str(market_actions):25} "
            f"stage={item['execution_stage']:22} "
            f"status={item['diagnostic_status']}"
        )

    # ------------------------------------------------------------------
    # 25. GENUINE FAILURE DETAILS
    # ------------------------------------------------------------------

    print("\nGENUINE FAILURE DETAILS")
    print("-" * 100)

    if not genuine_failures:
        print("No genuine diagnostic failures detected.")

    else:
        for item in genuine_failures[:15]:
            report = item["report"]
            record = item["record"]

            farmer_action = (
                _farmer_action(record)
                if record is not None
                else []
            )

            market_actions = (
                _market_actions(record)
                if record is not None
                else []
            )

            print(
                f"step={report.get('step'):4}\n"
                f"  decision:       "
                f"{report.get('decision_name')}\n"
                f"  farmer_action:  "
                f"{farmer_action}\n"
                f"  market_action:  "
                f"{market_actions}\n"
                f"  verification:   "
                f"{item['terminal_verification']}\n"
                f"  reason:         "
                f"{report.get('decision_reason')}\n"
                f"  expected:       "
                f"{report.get('expected_effect')}\n"
                f"  observed:       "
                f"{report.get('observed_effect')}\n"
                f"  agent_effect:   "
                f"{report.get('agent_effect')}\n"
                f"  environment:    "
                f"{report.get('environment_effect')}\n"
                f"  evidence:       "
                f"{report.get('evidence')}\n"
            )

    # ------------------------------------------------------------------
    # 26. MVP READINESS SNAPSHOT
    # ------------------------------------------------------------------

    completed = status_counts.get(
        "COMPLETED",
        0,
    )

    failed = status_counts.get(
        "FAILED",
        0,
    )

    navigation = status_counts.get(
        "IN_PROGRESS",
        0,
    )

    transitions = status_counts.get(
        "ENVIRONMENT_TRANSITION",
        0,
    )

    terminal_unconfirmed = status_counts.get(
        "TERMINAL_UNCONFIRMED",
        0,
    )

    target_verified = verification_counts.get(
        "TARGET_VERIFIED",
        0,
    )

    effect_verified = verification_counts.get(
        "EFFECT_VERIFIED",
        0,
    )

    print("\nMVP READINESS SNAPSHOT")
    print("-" * 100)

    print(
        f"Decision records:           "
        f"{total}"
    )

    print(
        f"Completed terminal outcomes:"
        f" {completed}"
    )

    print(
        f"Strategic progress records: "
        f"{navigation}"
    )

    print(
        f"Environment transitions:    "
        f"{transitions}"
    )

    print(
        f"Terminal unconfirmed:       "
        f"{terminal_unconfirmed}"
    )

    print(
        f"Target-verified terminals:  "
        f"{target_verified}"
    )

    print(
        f"Effect-verified terminals:  "
        f"{effect_verified}"
    )

    print(
        f"Genuine failures:           "
        f"{failed}"
    )

    print(
        f"Net profit:                 "
        f"{evaluation['net_profit']:.2f}"
    )

    print(
        f"ROI:                        "
        f"{evaluation['roi_percent']:.2f}%"
    )

    # ------------------------------------------------------------------
    # 27. MVP INTERPRETATION
    # ------------------------------------------------------------------

    print("\nMVP INTERPRETATION")

    if failed == 0:
        print(
            "  No genuine execution failures were detected "
            "by this diagnostic."
        )
    else:
        print(
            "  Genuine failures require investigation before "
            "being treated as an MVP-quality issue."
        )

    if navigation:
        print(
            "  Persistent strategic intents are producing "
            "multi-step execution progress."
        )

    if target_verified:
        print(
            "  Terminal spatial actions include target-level "
            "verification from persisted execution targets."
        )

    if effect_verified:
        print(
            "  Some terminal actions are verified by aggregate "
            "state effects where target coordinates are unavailable."
        )

    if terminal_unconfirmed:
        print(
            "  Some terminal actions remain unconfirmed because "
            "the available record does not prove their target-specific "
            "state transition."
        )

    if transitions:
        print(
            "  Environment-boundary records are isolated from "
            "ordinary decision failures."
        )

    print("\nIMPORTANT")

    print(
        "  RAW evaluator quality and strategic execution quality "
        "are reported separately."
    )

    print(
        "  RAW PARTIAL is not treated as a failure when the "
        "agent is demonstrably navigating or when terminal "
        "state evidence verifies the intended effect."
    )

    print(
        "  Harvest verification consumes target-level evidence "
        "from memory[*] outcome, captured by NexusAgentRunner."
    )

    print(
        "  This diagnostic does not modify the underlying "
        "DecisionOutcomeEvaluator."
    )

    print(
        "  Target-level weed verification remains limited because "
        "the current runner stores aggregate weed counts rather "
        "than weed coordinates."
    )

    print("=" * 100)

    return result


if __name__ == "__main__":
    for seed in DIAGNOSTIC_SEEDS:
        summarize_seed(seed)