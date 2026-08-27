"""Versioned models, so a bad retrain can be undone in one command.

Retraining is the standard response to drift, and retraining sometimes makes
things worse - a new period's data can be noisier, a feature can break upstream,
a label pipeline can silently change. Without versions the only options are to
live with it or to retrain again and hope.

Every version records the metrics it was accepted on. `promote` refuses to
activate a version that is materially worse than the current one, so a bad
retrain is caught at promotion time rather than in production - and `rollback`
exists for the case that gets through anyway.

Layout:

    artifacts/registry/
        registry.json           which version is active, and the full history
        v1/  v2/  v3/ ...       model files and the metrics for each
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# A new version may be this much worse before promotion is refused. Some
# variation between retrains is normal - the wire rail's seed spread on PR-AUC
# is 20% - so a threshold near zero would block every legitimate retrain.
REGRESSION_TOLERANCE = 0.05


@dataclass
class Version:
    version: str
    created: str
    rail: str
    metrics: dict
    notes: str = ""
    files: list[str] = field(default_factory=list)
    active: bool = False


class ModelRegistry:
    """Stores model versions and controls which one is live."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "registry.json"
        self.index = json.loads(self.index_path.read_text()) if self.index_path.exists() \
            else {"active": {}, "versions": []}

    def _save(self) -> None:
        self.index_path.write_text(json.dumps(self.index, indent=2))

    # -- writing ------------------------------------------------------------
    def register(self, rail: str, files: list[Path], metrics: dict,
                 notes: str = "") -> Version:
        """Store a new version. Registering does not activate it."""
        existing = [v for v in self.index["versions"] if v["rail"] == rail]
        version_id = f"v{len(existing) + 1}"
        folder = self.root / rail / version_id
        folder.mkdir(parents=True, exist_ok=True)

        stored = []
        for path in files:
            path = Path(path)
            shutil.copy2(path, folder / path.name)
            stored.append(path.name)

        version = Version(
            version=version_id,
            created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            rail=rail, metrics=metrics, notes=notes, files=stored,
        )
        (folder / "metrics.json").write_text(json.dumps(asdict(version), indent=2))
        self.index["versions"].append(asdict(version))
        self._save()
        return version

    def promote(self, rail: str, version_id: str, metric: str = "recall_at_operating_point",
                force: bool = False) -> dict:
        """Make a version live, refusing a clear regression.

        The comparison metric defaults to recall at the operating point rather
        than PR-AUC. Measured on the wire rail across three seeds: PR-AUC varied
        20.5% while recall at 1% varied 0.9%. Gating on the noisier number would
        reject good retrains and accept bad ones at random.
        """
        candidate = self._get(rail, version_id)
        if candidate is None:
            raise KeyError(f"{rail}/{version_id} is not registered")

        current_id = self.index["active"].get(rail)
        if current_id and not force:
            current = self._get(rail, current_id)
            before, after = current["metrics"].get(metric), candidate["metrics"].get(metric)
            if before is not None and after is not None:
                drop = (before - after) / before if before else 0.0
                if drop > REGRESSION_TOLERANCE:
                    return {"promoted": False, "reason":
                            f"{metric} would fall {drop:.1%} "
                            f"({before:.4f} -> {after:.4f}), over the "
                            f"{REGRESSION_TOLERANCE:.0%} tolerance",
                            "active": current_id}

        self.index["active"][rail] = version_id
        for v in self.index["versions"]:
            if v["rail"] == rail:
                v["active"] = v["version"] == version_id
        self._save()
        return {"promoted": True, "active": version_id,
                "previous": current_id, "reason": "forced" if force else "metrics acceptable"}

    def rollback(self, rail: str) -> dict:
        """Return to the version that was live before the current one."""
        history = [v for v in self.index["versions"] if v["rail"] == rail]
        current_id = self.index["active"].get(rail)
        if not current_id or len(history) < 2:
            return {"rolled_back": False, "reason": "no earlier version to return to"}

        order = [v["version"] for v in history]
        previous = order[order.index(current_id) - 1] if order.index(current_id) > 0 else None
        if previous is None:
            return {"rolled_back": False, "reason": "already on the earliest version"}

        # force=True by design: rolling back is a deliberate act, and the whole
        # point is that the metrics gate has already been overtaken by reality.
        self.promote(rail, previous, force=True)
        return {"rolled_back": True, "from": current_id, "to": previous}

    # -- reading ------------------------------------------------------------
    def _get(self, rail: str, version_id: str) -> dict | None:
        return next((v for v in self.index["versions"]
                     if v["rail"] == rail and v["version"] == version_id), None)

    def active(self, rail: str) -> dict | None:
        version_id = self.index["active"].get(rail)
        return self._get(rail, version_id) if version_id else None

    def path(self, rail: str, version_id: str | None = None) -> Path:
        version_id = version_id or self.index["active"].get(rail)
        return self.root / rail / version_id

    def history(self, rail: str | None = None) -> list[dict]:
        versions = self.index["versions"]
        return [v for v in versions if v["rail"] == rail] if rail else versions

    def describe(self, rail: str | None = None) -> str:
        lines = [f"{'rail':<10} {'version':<9} {'created':<22} "
                 f"{'recall@op':>10} {'active':>7}", "-" * 64]
        for v in self.history(rail):
            recall = v["metrics"].get("recall_at_operating_point", float("nan"))
            lines.append(f"{v['rail']:<10} {v['version']:<9} {v['created']:<22} "
                         f"{recall:>10.4f} {'  <--' if v['active'] else '':>7}")
        return "\n".join(lines)
