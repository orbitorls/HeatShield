from __future__ import annotations

import numpy as np
import pytest
from app.ml.safety.post_processor import SafetyPostProcessor


def test_noop_when_default_config():
    sp = SafetyPostProcessor("__UNKNOWN__", 6)
    hi_m, hi_l, hi_u, reason = sp.process(
        hi_mean=38.0, hi_lower=35.0, hi_upper=41.0,
        last_observed_hi=37.0,
    )
    assert hi_m == 38.0
    assert reason is None


def test_noop_without_last_observed():
    sp = SafetyPostProcessor("BKK_01", 6)
    hi_m, hi_l, hi_u, r = sp.process(
        hi_mean=38.0, hi_lower=35.0, hi_upper=41.0,
    )
    assert hi_m == 38.0
    assert r is None


def test_persistence_blend_bkk_h6():
    sp = SafetyPostProcessor("BKK_01", 6)
    hi_m, hi_l, hi_u, r = sp.process(
        hi_mean=40.5, hi_lower=38.0, hi_upper=43.0,
        last_observed_hi=39.0,
    )
    assert hi_m < 40.5
    assert "blend_w" in (r or "")


def test_danger_boost_ryg_h6():
    sp = SafetyPostProcessor("RYG_01", 6)
    hi_m, hi_l, hi_u, r = sp.process(
        hi_mean=39.0, hi_lower=36.0, hi_upper=42.0,
    )
    assert hi_m > 39.0
    assert "danger_boost" in (r or "")


def test_no_boost_below_warning_threshold():
    sp = SafetyPostProcessor("RYG_01", 6)
    hi_m, hi_l, hi_u, r = sp.process(
        hi_mean=35.0, hi_lower=32.0, hi_upper=38.0,
    )
    assert hi_m == 35.0
    assert r is None


def test_boost_clamped_to_upper_pi():
    sp = SafetyPostProcessor("RYG_01", 6)
    hi_m, hi_l, hi_u, r = sp.process(
        hi_mean=40.0, hi_lower=38.0, hi_upper=40.8,
    )
    assert hi_m <= 40.8


def test_pi_sanity_clamp_wide():
    sp = SafetyPostProcessor("BKK_01", 6)
    hi_m, hi_l, hi_u, r = sp.process(
        hi_mean=38.0, hi_lower=10.0, hi_upper=55.0,
    )
    assert abs(hi_u - hi_m - 5.0) < 0.01
    assert abs(hi_m - hi_l - 5.0) < 0.01
    assert "pi_clamped" in (r or "")


def test_pi_lower_greater_than_mean():
    sp = SafetyPostProcessor("BKK_01", 6)
    hi_m, hi_l, hi_u, r = sp.process(
        hi_mean=38.0, hi_lower=39.0, hi_upper=42.0,
    )
    assert hi_l < hi_m
    assert "pi_lower>mean" in (r or "")


def test_pi_upper_less_than_mean():
    sp = SafetyPostProcessor("BKK_01", 6)
    hi_m, hi_l, hi_u, r = sp.process(
        hi_mean=38.0, hi_lower=35.0, hi_upper=37.0,
    )
    assert hi_u > hi_m
    assert "pi_upper<mean" in (r or "")


def test_blend_plus_boost_chain():
    sp = SafetyPostProcessor("RYG_01", 6)
    hi_m, hi_l, hi_u, r = sp.process(
        hi_mean=40.5, hi_lower=37.0, hi_upper=44.0,
        last_observed_hi=38.0,
    )
    assert hi_m >= 38.0
    assert r is not None
    assert "blend" in r or "danger_boost" in r


def test_default_config_for_unknown_station():
    sp = SafetyPostProcessor("NONEXISTENT_01", 24)
    assert sp.config.blend_weight == 1.0
    assert sp.config.danger_boost == 0.0


def test_ryg_h6_has_boost():
    cfg = SafetyPostProcessor("RYG_01", 6).config
    assert cfg.danger_boost > 0.0


def test_bkk_h6_has_blend():
    cfg = SafetyPostProcessor("BKK_01", 6).config
    assert cfg.blend_weight < 1.0


def test_enabled_false_horizon_still_processes():
    sp = SafetyPostProcessor("KKN_01", 48)
    assert sp.config.enabled is False
    hi_m, hi_l, hi_u, r = sp.process(
        hi_mean=42.0, hi_lower=38.0, hi_upper=46.0,
        last_observed_hi=39.0,
    )
    assert r is not None


def test_unknown_horizon_no_modifications():
    sp = SafetyPostProcessor("CNX_01", 99)
    hi_m, hi_l, hi_u, r = sp.process(
        hi_mean=42.0, hi_lower=38.0, hi_upper=46.0,
        last_observed_hi=39.0,
    )
    assert r is None
    assert hi_m == 42.0
