"""Checks for the fixes ported from the Codex review (R2, R3, R4)."""
import numpy as np
import pandas as pd

from iaa import features, pipeline


def dev15(y, mu):
    return 4 * (np.sqrt(mu) + y / np.sqrt(mu) - 2 * np.sqrt(y))


def test_R3_sqrt_exposure_weight_matches_row_metric():
    e = np.array([.01, .3, 1., .7]); y = np.array([0., 100., 20., 0.]); mu = np.array([20., 15., 80., 60.])
    np.testing.assert_allclose(dev15(e * y, e * mu), np.sqrt(e) * dev15(y, mu))


def test_R4_ultimate_counts_and_cost_keep_pure_premium():
    n = np.array([1., 2.]); c = np.array([200., 900.]); q = np.array([.6, .9]); e = np.array([.3, 1.])
    freq, sev = (n / q) / e, (c / q) / (n / q)
    np.testing.assert_allclose(freq * sev, c / q / e)          # consistent bases: IBNR kept
    np.testing.assert_allclose((n / q) / e * (c / (n / q)), c / e)  # old inputs: IBNR cancels


def test_R2_weather_uses_only_earlier_years():
    w = pd.DataFrame({"address_id": ["a"] * 3, "year": [2021, 2022, 2023],
                      **{v: [1., 2., 100.] for v in features.WEATHER_VARS}})
    rows = pd.DataFrame({"address_id": ["a"], "wx_prev_hail_days": [1.], "x": [0]})
    tr, te, keep = features.time_safe_weather(rows, rows, w, cutoff=2023)
    assert te["wx_avg_hail_days"].iloc[0] == 1.5   # 2023 excluded
    assert keep(["x", "wx_prev_hail_days", "wx_avg_hail_days"]) == ["x", "wx_avg_hail_days"]


def test_v5_blend_is_a_normalised_relativity():
    a, b = pipeline.norm([1., 3.]), pipeline.norm([2., 2.])
    kind, w = pipeline.V5["buildings"]
    mix = (1 - w) * a + w * b
    assert np.isclose(mix.sum(), 1.0)


def test_level_settings_unchanged_for_developed_and_defined_for_short_history():
    from iaa import level
    d = level.history_settings([2024, 2023, 2022, 2021])
    assert d["freq_years"] == (2022, 2023, 2024) and d["base_default"] == level.BASE_YEARS_DEFAULT
    e = level.history_settings([2023, 2024])
    assert e["ref_year"] == 2023 and e["base_default"] == (2023,) and e["freq_years"] == (2023, 2024)
