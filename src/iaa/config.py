"""Central settings, from the Kaggle Data tab (16 Oct 2026)."""

import pandas as pd

SEED = 20261016
TWEEDIE_POWER = 1.5  # competition metric power; also used for model loss

LINES = ("motor", "buildings", "contents")
ID_PREFIX = {"motor": "M", "buildings": "B", "contents": "C"}  # C is assumed, confirm on test_contents
VALUATION_DATE = pd.Timestamp("2024-12-31")  # claims are valued at this date
ID_COLS = ("customer_id", "address_id", "policy_id", "row_id", "claim_id")  # always read as text

COLS = {
    "id": "row_id",
    "year": "policy_year",       # derived from period_start
    "exposure": "exposure",
    "target": "claim_cost",      # built in target.py: matched, ex repudiated, net of recoveries, to ultimate
}

PERILS = {
    "motor": ("own_damage", "third_party_prop", "bodily_injury", "theft", "glass_hail"),
    "buildings": ("storm", "flood", "escape_of_water", "subsidence", "fire", "accidental_damage"),
    "contents": ("theft", "escape_of_water", "accidental_damage", "flood"),
}

# ---------------------------------------------------------------------------
# Feature policy. Every exclusion has a reason; this table goes into the write-up.
# ---------------------------------------------------------------------------
EXCLUDE = {
    # Leakage: information not known when the 2025 price is set
    "renewal_decision": "leakage: outcome at end of period; training files only",
    "customer_service_contacts": "leakage risk: may include contacts about claims in the same period; test before any use",
    # Identifiers and dates (replaced by derived fields)
    "row_id": "identifier", "policy_id": "identifier", "customer_id": "identifier",
    "address_id": "identifier", "period_start": "date (used to derive year/season)",
    "period_end": "date", "date_of_birth": "replaced by age at period start",
    # Fairness and proxy discrimination (FCA Consumer Duty; proxy for protected characteristics)
    "surname": "fairness: potential proxy for ethnicity; no legitimate risk rationale",
    "email_domain": "fairness: proxy for age and socio-economic group, no causal link to claims",
    # Marketing and admin fields with no risk rationale; likely noise or proxies
    "marketing_segment": "no risk rationale; assigned by marketing, may encode protected traits",
    "campaign_code": "no risk rationale; acquisition artefact",
    "marketing_consent": "no risk rationale",
    "document_preference": "no risk rationale; proxy for age",
    "branch_id": "no risk rationale; admin artefact",
    # Pricing circularity
    "written_premium": "circular: encodes the insurer's current pricing; target is risk premium. Test as challenger only",
    "discount_applied": "circular: a pricing decision, not a risk characteristic",
}

# Kept, but reviewed for fairness and documented (permitted rating factors in UK practice,
# but each needs a stated risk rationale and a proxy check in the write-up).
REVIEW = ("occupation", "credit_band", "marital_status", "previous_insurer",
          "missed_payments_12m", "vehicle_colour", "household_size")

# Claims-table fields and anything recorded after a loss: never a feature.
POST_LOSS_FIELD_HINTS = (
    "claim", "loss_date", "report_date", "paid", "incurred", "reserve", "recover",
    "settle", "repudiat", "status", "repairer", "third_party_id", "renewal_decision",
)
