"""Central settings. Change column names here once the Data tab is confirmed."""

SEED = 20261016
TWEEDIE_POWER = 1.5  # competition metric power; also used for model loss

LINES = ("motor", "buildings", "contents")
ID_PREFIX = {"motor": "M", "buildings": "B", "contents": "C"}  # C is assumed, confirm

# Placeholder column names. Replace after reading the Kaggle Data tab.
COLS = {
    "id": "row_id",
    "year": "policy_year",
    "exposure": "exposure",          # earned years in the row's period; confirm it exists
    "target": "claim_cost",          # ultimate, net of excess and recoveries, ex declined
}

# Fields that only exist after a loss happens. Any of these in the feature set = leakage.
# Extend once the real schema is known.
POST_LOSS_FIELD_HINTS = (
    "claim", "loss_date", "paid", "incurred", "reserve", "recovery",
    "settle", "declined", "status", "excess_applied", "report_date",
)
