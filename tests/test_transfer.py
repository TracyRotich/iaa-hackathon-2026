import pandas as pd
from iaa import transfer


def test_harmonise_keeps_ids_and_maps_age_bands():
    df = pd.DataFrame({"row_id": ["M001"], "driver_age": [24], "cover_type": [" Comprehensive "]})
    h = transfer.harmonise(df, "motor")
    assert h.row_id.iloc[0] == "M001"                 # IDs must never be altered (bug caught 28 Oct)
    assert h.driver_age_band.iloc[0] == "17-24"
    assert h.cover_type.iloc[0] == "comprehensive"
    assert transfer.harmonise(pd.DataFrame({"driver_age": [25, 64, 65]}), "motor").driver_age_band.tolist() == ["25-34", "50-64", "65+"]


def test_sum_insured_unit_repair():
    df = pd.DataFrame({"sum_insured": [250_000.0, 303_000_000.0]})
    r = transfer.repair_sum_insured(df, "buildings")
    assert r.sum_insured.tolist() == [250_000.0, 303_000.0] and r.si_unit_repaired.tolist() == [0, 1]
    assert r.sum_insured_raw.tolist() == df.sum_insured.tolist()


def test_calibration_applies_mix_once_and_ignores_model_scale():
    import numpy as np
    rng = np.random.default_rng(1)
    e_tr, e_te = rng.uniform(.2, 1, 500), rng.uniform(.2, 1, 300)
    q_tr, q_te = rng.lognormal(0, .5, 500), rng.lognormal(.2, .5, 300)
    p_tr, p_te = q_tr * e_tr, q_te * e_te
    pred, mix = transfer.calibrate_to_level(p_tr, e_tr, p_te, e_te, level=100.0)
    # equivalent form: test relativities rescaled to level x mix x exposure
    alt = p_te / p_te.sum() * 100.0 * mix * e_te.sum()
    np.testing.assert_allclose(pred, alt)
    # scaling the model by any constant leaves predictions unchanged
    np.testing.assert_allclose(transfer.calibrate_to_level(7 * p_tr, e_tr, 7 * p_te, e_te, 100.0)[0], pred)
    # same portfolio in train and test gives mix = 1 and the level exactly
    pred2, mix2 = transfer.calibrate_to_level(p_tr, e_tr, p_tr, e_tr, 100.0)
    assert abs(mix2 - 1) < 1e-12 and abs(pred2.sum() / e_tr.sum() - 100.0) < 1e-9
