import numpy as np
import pandas as pd
import pytest

from iaa import target
from iaa.config import EXCLUDE, POST_LOSS_FIELD_HINTS
from iaa.guards import leakage_suspects

T = pd.Timestamp


@pytest.fixture
def toy():
    # Policy P1 changes mid-year (two rows); P2 is cancelled at end of March 2024.
    policies = pd.DataFrame({
        "row_id": ["M1", "M2", "M3"],
        "policy_id": ["P1", "P1", "P2"],
        "period_start": [T("2024-01-01"), T("2024-07-01"), T("2024-01-01")],
        "period_end": [T("2024-06-30"), T("2024-12-31"), T("2024-03-31")],
        "exposure": [0.5, 0.5, 0.25],
    })
    claims = pd.DataFrame({
        "claim_id": ["C1", "C2", "C3", "C4", "C5"],
        "policy_id": ["P1", "P1", "P1", "P2", "P2"],
        "loss_date": [T("2024-03-10"), T("2024-08-01"), T("2024-09-01"), T("2024-02-01"), T("2024-05-01")],
        "report_date": [T("2024-03-12"), T("2024-08-03"), T("2024-09-05"), T("2024-02-02"), T("2024-05-02")],
        "status": ["closed", "open", "repudiated", "closed", "closed"],
        "paid_to_date": [1000.0, 200.0, 0.0, 500.0, 300.0],
        "incurred_to_date": [1000.0, 800.0, 400.0, 500.0, 300.0],
        "recoveries": [100.0, 0.0, 0.0, 0.0, 0.0],
    })
    return policies, claims


def test_claims_go_to_row_in_force_on_loss_date(toy):
    pol, clm = toy
    matched, unmatched = target.match_claims(pol, clm)
    got = dict(zip(matched["claim_id"], matched["row_id"]))
    assert got == {"C1": "M1", "C2": "M2", "C3": "M2", "C4": "M3"}
    assert list(unmatched["claim_id"]) == ["C5"]  # loss after cancellation: reported, not kept


def test_value_drops_repudiated_and_nets_recoveries(toy):
    pol, clm = toy
    matched, _ = target.match_claims(pol, clm)
    rows = target.reported_cost_by_row(pol, target.claim_value(matched)).set_index("row_id")
    assert rows.loc["M1", "reported_cost"] == 900.0      # 1000 incurred - 100 recovered
    assert rows.loc["M2", "reported_cost"] == 800.0      # repudiated C3 excluded
    assert rows.loc["M2", "n_claims"] == 1


def test_ibner_factor_applies_to_open_only(toy):
    pol, clm = toy
    matched, _ = target.match_claims(pol, clm)
    v = target.claim_value(matched, ibner_factor=1.1).set_index("claim_id")["value"]
    assert v["C2"] == pytest.approx(880.0) and v["C1"] == 900.0


def test_completeness_is_lower_for_recent_rows():
    rng = np.random.default_rng(1)
    n = 4000
    loss = T("2021-01-01") + pd.to_timedelta(rng.integers(0, 1460, n), unit="D")
    delay = rng.exponential(60, n).astype(int)
    claims = pd.DataFrame({"loss_date": loss, "report_date": loss + pd.to_timedelta(delay, unit="D")})
    grid, cdf = target.reporting_delay_cdf(claims, max_delay_days=365)
    pol = pd.DataFrame({"period_start": [T("2022-01-01"), T("2024-10-01")],
                        "period_end": [T("2022-12-31"), T("2024-12-31")]})
    comp = target.row_completeness(pol, grid, cdf)
    assert comp[0] > 0.99 and comp[1] < 0.6  # old row complete; Q4 2024 row heavily IBNR


def test_policy_excludes_leakage_and_fairness_fields():
    for f in ("renewal_decision", "surname", "email_domain", "written_premium"):
        assert f in EXCLUDE
    assert "renewal_decision" in leakage_suspects(["renewal_decision", "ncb_years"])
    assert leakage_suspects(["ncb_years", "driver_age", "flood_zone", "prior_claims_5y"]) == []


def test_duplicate_claims_dropped(toy):
    _, clm = toy
    dup = clm.iloc[[0]].assign(claim_id="C999999999")
    out, n = target.drop_duplicate_claims(pd.concat([clm, dup]))
    assert n == 1 and len(out) == len(clm)


def test_exposure_repaired_from_dates():
    from iaa.load import repair_exposure
    df = pd.DataFrame({"period_start": [T("2021-01-01"), T("2021-02-08"), T("2021-07-01")],
                       "period_end": [T("2021-12-31"), T("2021-12-31"), T("2021-12-31")],
                       "exposure": [1.0139, 327.0, 184 / 365]})  # 360-day basis, days, correct
    out, n = repair_exposure(df)
    assert n == 2 and out["exposure"].max() <= 1.0 and out["exposure"].iloc[2] == pytest.approx(184 / 365)


def test_marital_status_not_flagged_as_leakage():
    assert leakage_suspects(["marital_status", "claim_status"]) == ["claim_status"]
