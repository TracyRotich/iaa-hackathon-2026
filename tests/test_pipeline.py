import numpy as np
import pandas as pd
import pytest

from iaa import guards, metrics, submission, synthetic, validation
from iaa.config import COLS
from iaa.models import TweedieGBM, TweedieGLM

FEATURES = ["driver_age", "region"]


@pytest.fixture(scope="module")
def data():
    return synthetic.make()


def test_metric_matches_baseline_definition(data):
    y = data[COLS["target"]]
    assert metrics.d2(y, np.full(len(y), y.mean())) == pytest.approx(0.0, abs=1e-9)


def test_models_beat_baseline_and_stay_calibrated(data):
    train, test = validation.time_split(data)
    table, preds = validation.evaluate(
        {"glm": TweedieGLM(), "gbm": TweedieGBM(n_estimators=200)}, train, test, FEATURES)
    assert (table["d2_tweedie_1.5"] > 0).all(), table
    # No trend adjustment yet, so allow a loose band; tighten once trend is modelled.
    assert table["pred_total / actual_total"].between(0.75, 1.25).all(), table
    for p in preds.values():
        guards.check_predictions(p)


def test_rerun_is_reproducible(data):
    train, test = validation.time_split(data)
    a = TweedieGBM(n_estimators=50).fit(train[FEATURES], train[COLS["target"]], train[COLS["exposure"]])
    b = TweedieGBM(n_estimators=50).fit(train[FEATURES], train[COLS["target"]], train[COLS["exposure"]])
    np.testing.assert_allclose(a.predict(test[FEATURES], test[COLS["exposure"]]),
                               b.predict(test[FEATURES], test[COLS["exposure"]]))


def test_leakage_guard_flags_post_loss_fields():
    feats = ["driver_age", "region", "claim_count", "paid_to_date", "vehicle_value"]
    assert set(guards.leakage_suspects(feats)) == {"claim_count", "paid_to_date"}


def test_exposure_guard(data):
    assert guards.check_exposure(data) == {}
    bad = data.copy()
    bad.loc[0, COLS["exposure"]] = 0
    assert "non_positive" in guards.check_exposure(bad)


def test_negative_predictions_rejected():
    with pytest.raises(ValueError):
        guards.check_predictions([1.0, -0.1])


def test_submission_validation():
    ids = ["M1", "B1", "C1"]
    sub = submission.build({
        "motor": pd.DataFrame({"row_id": ["M1"], "prediction": [10.0]}),
        "buildings": pd.DataFrame({"row_id": ["B1"], "prediction": [5.0]}),
        "contents": pd.DataFrame({"row_id": ["C1"], "prediction": [0.0]}),
    })
    assert submission.validate(sub, ids)
    with pytest.raises(ValueError):
        submission.validate(sub.iloc[:2], ids)
