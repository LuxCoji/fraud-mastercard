"""Assemble the operations console: template plus exported data, inlined.

The page is deliberately static and self-contained. An investigator console that
needs a backend to render is one more thing that can be down at 3am, and the
expensive work - building the account graph, tracing flows - has already
happened in `export_dashboard_data.py`.

Run:
    python src/models/export_dashboard_data.py
    python src/models/build_dashboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg

MARKER = "/*__DATA__*/null"

TEMPLATE = cfg.ROOT / "deploy" / "console.template.html"
OUTPUT = cfg.ROOT / "deploy" / "console.html"


def main() -> None:
    # Read the exporter's own output. An earlier version read a `.min.json`
    # that no script produced, so a clean checkout raised FileNotFoundError and
    # a stale copy silently rebuilt the console from old data.
    source = cfg.REPORTS / "dashboard_data.json"
    if not source.exists():
        raise SystemExit(
            f"{source.name} not found - run export_dashboard_data.py first")

    data = source.read_text(encoding="utf-8")
    template = TEMPLATE.read_text(encoding="utf-8")

    # Assert the substitution happened. Without this the script writes a page
    # containing `const DATA = null`, prints a success line, and the console
    # renders blank with an error only visible in the browser devtools.
    if template.count(MARKER) != 1:
        raise SystemExit(
            f"expected exactly one {MARKER!r} in the template, "
            f"found {template.count(MARKER)}")

    OUTPUT.write_text(template.replace(MARKER, data), encoding="utf-8")
    print(f"{OUTPUT.relative_to(cfg.ROOT)}  {OUTPUT.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
