"""Download the active model artifacts from GCS before the server starts.

Model files are deliberately not in the repository. They are 40 MB of boosters
that change independently of the code, and git history cannot be cleaned once
they are in it. Keeping them in a bucket means:

  - a clone stays small and CI stays fast
  - promoting a model is an upload, not a commit and rebuild
  - `ModelRegistry.rollback` has something real to point at

Set MODEL_BUCKET to enable. Without it this exits quietly, so the image still
runs locally against whatever is in artifacts/.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"

# Only what serving needs. The registry, scores and reports stay in the bucket.
REQUIRED = [
    "card_lightgbm.txt",
    "wire_xgboost_medium.json",
    "wire_feature_names_medium.json",
    "wire_model_card_medium.json",
]


def main() -> None:
    bucket = os.environ.get("MODEL_BUCKET", "").strip()
    if not bucket:
        print("MODEL_BUCKET unset - using whatever is already in artifacts/")
        return

    from google.cloud import storage

    prefix = os.environ.get("MODEL_PREFIX", "fraud-mastercard/artifacts").strip("/")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    client = storage.Client()

    missing = []
    for name in REQUIRED:
        blob = client.bucket(bucket).blob(f"{prefix}/{name}")
        if not blob.exists(client):
            missing.append(name)
            continue
        blob.download_to_filename(ARTIFACTS / name)
        print(f"fetched {name} ({(ARTIFACTS / name).stat().st_size / 1e6:.1f} MB)")

    if missing:
        # Fail loudly. A server that starts without its models will happily
        # return "allow, unscored" for every transaction, which is worse than
        # not starting.
        print(f"MISSING from gs://{bucket}/{prefix}/: {missing}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
