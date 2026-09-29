import pytest

from nexus.intelligence.risk.models import CropRisk


def test_crop_risk_defaults_to_low_risk():
    risk = CropRisk(crop="WHEAT")

    assert risk.weather == 0.0
    assert risk.disease == 0.0
    assert risk.pest == 0.0
    assert risk.market == 0.0
    assert risk.operational == 0.0
    assert risk.overall == 0.0


def test_crop_risk_calculates_overall_risk():
    risk = CropRisk(
        crop="MELON",
        weather=0.8,
        disease=0.6,
        pest=0.4,
        market=0.2,
        operational=0.5,
    )

    assert risk.overall == pytest.approx(0.5)


def test_crop_risk_rejects_invalid_values():
    with pytest.raises(ValueError):
        CropRisk(crop="WHEAT", weather=1.1)

    with pytest.raises(ValueError):
        CropRisk(crop="WHEAT", disease=-0.1)


def test_crop_risk_supports_weighted_aggregation():
    risk = CropRisk(
        crop="MELON",
        weather=0.8,
        disease=0.2,
        pest=0.3,
        market=0.1,
        operational=0.2,
    )

    weights = {
        "weather": 0.35,
        "disease": 0.25,
        "pest": 0.15,
        "market": 0.15,
        "operational": 0.10,
    }

    assert risk.weighted_overall(weights) == pytest.approx(0.41)


def test_crop_risk_rejects_negative_weights():
    risk = CropRisk(crop="MELON", weather=0.5)

    weights = {
        "weather": -0.35,
        "disease": 0.25,
        "pest": 0.15,
        "market": 0.15,
        "operational": 0.10,
    }

    with pytest.raises(ValueError, match="cannot be negative"):
        risk.weighted_overall(weights)


def test_crop_risk_rejects_zero_total_weight():
    risk = CropRisk(crop="MELON", weather=0.5)

    weights = {
        "weather": 0.0,
        "disease": 0.0,
        "pest": 0.0,
        "market": 0.0,
        "operational": 0.0,
    }

    with pytest.raises(ValueError, match="greater than zero"):
        risk.weighted_overall(weights)


def test_operational_risk_detects_unwatered_crop():
    from nexus.core.state import CropState, FarmState
    from nexus.intelligence.risk.signals import operational_risk

    crop = CropState(
        crop="WHEAT",
        row=0,
        col=0,
        planted_day=0,
        watered_today=False,
        yield_units=0,
        fertilized_until_day=-1,
    )

    state = FarmState(
        player_id=0,
        day=1,
        hour=0,
        farmer_position=[0, 0],
        crops=[crop],
    )

    assert operational_risk(crop, state) == 0.5


def test_operational_risk_combines_observed_concerns():
    from nexus.core.state import CropState, FarmState
    from nexus.intelligence.risk.signals import operational_risk

    crop = CropState(
        crop="WHEAT",
        row=0,
        col=0,
        planted_day=0,
        watered_today=False,
        yield_units=2,
        fertilized_until_day=1,
    )

    state = FarmState(
        player_id=0,
        day=4,
        hour=0,
        farmer_position=[0, 0],
        crops=[crop],
    )

    assert operational_risk(crop, state) == 1.0

def test_operational_risk_is_zero_when_no_concern_is_observed():
    from nexus.core.state import CropState, FarmState
    from nexus.intelligence.risk.signals import operational_risk

    crop = CropState(
        crop="WHEAT",
        row=0,
        col=0,
        planted_day=0,
        watered_today=True,
        yield_units=0,
        fertilized_until_day=-1,
    )

    state = FarmState(
        player_id=0,
        day=1,
        hour=0,
        farmer_position=[0, 0],
        crops=[crop],
    )

    assert operational_risk(crop, state) == 0.0

def test_risk_signal_builder_builds_operational_crop_risk():
    from nexus.core.state import CropState, FarmState
    from nexus.intelligence.risk.builder import RiskSignalBuilder

    crop = CropState(
        crop="WHEAT",
        row=0,
        col=0,
        planted_day=0,
        watered_today=False,
        yield_units=0,
        fertilized_until_day=-1,
    )

    state = FarmState(
        player_id=0,
        day=1,
        hour=0,
        farmer_position=[0, 0],
        crops=[crop],
    )

    risk = RiskSignalBuilder().build(crop, state)

    assert risk.crop == "WHEAT"
    assert risk.operational == 0.5
    assert risk.weather == 0.0
    assert risk.disease == 0.0
    assert risk.pest == 0.0
    assert risk.market == 0.0
