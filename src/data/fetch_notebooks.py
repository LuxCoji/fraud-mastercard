"""Pull the reference IEEE-CIS kernels.

Kaggle's kernels/list and competitions/list endpoints return 401 for this
account, but kernels/pull by explicit ref works, so the refs are pinned here
rather than discovered by search.

These are the card-rail references only. PaySim and IBM AML have no public
notebook worth copying; those heads are built from scratch.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import requests
from requests.auth import HTTPBasicAuth

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg

DEST = cfg.ROOT / "notebooks" / "reference"

# (user, slug, why it is here)
REFS = [
    ("cdeotte", "xgb-fraud-with-magic-0-9600",
     "XGBoost, 0.9600 public LB. Source of the UID 'magic feature'."),
    ("kyakovlev", "ieee-lgbm-with-groupkfold-cv",
     "LightGBM params + GroupKFold. Closest to our intended setup."),
    ("kyakovlev", "ieee-fe-with-some-eda",
     "Feature engineering: normalisation, frequency and aggregation encoding."),
    ("cdeotte", "eda-for-columns-v-and-id",
     "What the V and id_ columns actually mean - the block we must recover."),
]


def main() -> None:
    creds = json.load(open(Path.home() / ".kaggle" / "kaggle.json"))
    auth = HTTPBasicAuth(creds["username"], creds["key"])
    DEST.mkdir(parents=True, exist_ok=True)

    manifest = []
    for user, slug, why in REFS:
        r = requests.get(
            "https://www.kaggle.com/api/v1/kernels/pull",
            params={"user_name": user, "kernel_slug": slug},
            auth=auth, timeout=120,
        )
        r.raise_for_status()
        payload = r.json()
        meta, blob = payload.get("metadata", {}), payload.get("blob", {})

        out = DEST / f"{user}__{slug}.ipynb"
        out.write_text(blob.get("source", ""), encoding="utf-8")

        manifest.append({
            "ref": f"{user}/{slug}",
            "title": meta.get("title"),
            "language": meta.get("language"),
            "votes": meta.get("totalVotes"),
            "file": out.name,
            "why": why,
        })
        print(f"saved {out.name:52s} {out.stat().st_size / 1e3:>8.1f} KB")

    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nmanifest -> {DEST / 'manifest.json'}")


if __name__ == "__main__":
    main()
