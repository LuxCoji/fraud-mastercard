"""Central paths and per-rail constants.

One place for filesystem layout so no module hardcodes a path.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"
ARTIFACTS = ROOT / "artifacts"

for _d in (RAW, INTERIM, PROCESSED, REPORTS, ARTIFACTS):
    _d.mkdir(parents=True, exist_ok=True)

# --- Raw source files -------------------------------------------------------
IEEE_TRANSACTION = RAW / "train_transaction.csv"
IEEE_IDENTITY = RAW / "train_identity.csv"
PAYSIM = RAW / "PS_20174392719_1491204439457_log.csv"
AML_TRANS = RAW / "HI-Small_Trans.csv"
AML_PATTERNS = RAW / "HI-Small_Patterns.txt"

# --- Rails ------------------------------------------------------------------
# A rail is a payment type with its own feature space, model and threshold.
# Routing between them is deterministic from the request payload, never learned.
RAILS = ("card", "transfer", "wire")

RAIL_SOURCE = {
    "card": "ieee_cis",      # card-not-present e-commerce authorisation
    "transfer": "paysim",    # P2P mobile-money push payment
    "wire": "ibm_aml",       # interbank wire / laundering
}

# Label column per source, as it appears in the raw file.
LABEL_COLUMN = {
    "ieee_cis": "isFraud",
    "paysim": "isFraud",
    "ibm_aml": "Is Laundering",
}

# --- Validation -------------------------------------------------------------
# Fraction of the time-ordered stream held out. Splits are always chronological
# and grouped by entity: a random split leaks the same card into train and test
# and inflates every metric downstream.
TEST_FRACTION = 0.20
CALIBRATION_FRACTION = 0.10

RANDOM_SEED = 42
