"""Consensus triage: N independent LLM reviews of a case, then a majority vote.

The wire rail queues cases at roughly 12% precision, so an investigator opening
the queue works through about eight false positives for every real laundering
operation. That is normal for AML and it is still the largest cost in the system.

This sits between the model and the investigator. It does not re-score anything;
it reads the case the model already produced - the traced flows, the amounts, the
timing - and asks whether the structure actually looks like laundering.

**The pattern comes from Visa's Vulnerability Agentic Harness** (Apache 2.0),
which uses majority-vote false-positive filtering: N independent passes at
non-zero temperature, and a finding is kept only if enough of them agree. The
reason it works is that a single LLM pass is confidently wrong in *different*
ways each time, so agreement is evidence and disagreement is a signal to escalate
rather than to decide.

Three rules this follows, and each exists because the alternative is worse:

**Never auto-dismiss.** A case the panel rejects is downgraded, not deleted. The
model flagged it for a measured reason and an LLM's disagreement is weaker
evidence than the model's score, not stronger.

**Disagreement escalates.** A split panel means the case is genuinely ambiguous,
which is exactly the case a human should see - not one to quietly resolve by
majority.

**The LLM never sees a decision it could bias.** It gets the structure and the
numbers, not the model's score, so its judgement is independent rather than an
echo of what it was told to agree with.
"""
from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# gpt-oss-120b is the strongest model available on this account. The panel runs
# off the critical path, so capability matters more than latency here.
DEFAULT_MODEL = "openai/gpt-oss-120b"

# Odd number so a majority always exists. Three is the smallest panel where one
# outlier cannot carry the vote.
PANEL_SIZE = 3

# Non-zero, and deliberately so. At temperature 0 the passes are near-identical
# and the vote measures nothing - the whole point is independent errors.
TEMPERATURE = 0.7

VERDICTS = ("laundering", "unclear", "legitimate")

SYSTEM_PROMPT = """You are reviewing a flagged case from a bank's anti-money-laundering system.

You will be shown the traced money flow - which accounts, how much, over what
period - and asked whether the structure is consistent with laundering.

Known laundering structures:
- CYCLE: money returns to its origin through intermediaries
- LAYERING (stack): a chain of hops moving most of the value onward
- FAN-OUT / FAN-IN: one account paying or collecting from many, in a burst
- SCATTER-GATHER: money split across intermediaries then reconverging

What is NOT laundering, and is the common false positive here:
- ordinary business paying many suppliers on a schedule
- payroll, which fans out but repeats predictably and does not reconverge
- a treasury account moving funds between its own entities
- a merchant settling with a payment processor
- high volume alone; large amounts alone; a single unusual transaction

Judge the STRUCTURE, not the size. A large legitimate payment is not laundering
and a small circular flow may be.

Reply with JSON only:
{"verdict": "laundering" | "unclear" | "legitimate",
 "confidence": 0.0-1.0,
 "reason": "one sentence"}"""


@dataclass
class Review:
    """One panel member's opinion."""

    verdict: str
    confidence: float
    reason: str
    error: str | None = None


@dataclass
class TriageResult:
    """What the panel decided, and how divided it was."""

    case_id: int
    verdict: str
    agreement: float                       # share of the panel that agreed
    reviews: list[Review] = field(default_factory=list)
    escalate: bool = False
    summary: str = ""

    @property
    def unanimous(self) -> bool:
        return self.agreement == 1.0

    def to_dict(self) -> dict:
        return {"case_id": self.case_id, "verdict": self.verdict,
                "agreement": round(self.agreement, 3), "escalate": self.escalate,
                "summary": self.summary,
                "reviews": [{"verdict": r.verdict, "confidence": r.confidence,
                             "reason": r.reason} for r in self.reviews]}


def load_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key
    env = Path(__file__).resolve().parents[2] / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("GROQ_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("GROQ_API_KEY not set and not found in .env")


def describe_case(case) -> str:
    """The case as the panel sees it.

    Deliberately excludes the model's score. The panel's job is an independent
    read of the structure; telling it the model was 0.99 confident would produce
    agreement with the model rather than a second opinion.
    """
    lines = [
        f"Alerts in this case: {len(case.transactions)}",
        f"Accounts involved: {len(case.accounts)}",
        f"Total value: {case.total_value:,.2f}",
    ]

    if case.flows:
        best = sorted(case.flows, key=lambda f: -f.hops)[:5]
        lines.append(f"\nTraced money flows ({len(case.flows)} found, showing {len(best)}):")
        for flow in best:
            shape = "CYCLE - returns to origin" if flow.is_cycle else "CHAIN"
            lines.append(
                f"  {shape}, {flow.hops} hops over {flow.span_hours:.0f} hours: "
                f"{' -> '.join(flow.accounts)}")
            lines.append(
                f"    {flow.amounts[0]:,.0f} in, {flow.amounts[-1]:,.0f} out "
                f"({flow.retained:.0%} retained)")

    if case.fans:
        widest = max(case.fans, key=lambda f: len(f.counterparties))
        lines.append(
            f"\nFan-{widest.direction}: {widest.hub} with "
            f"{len(widest.counterparties)} counterparties, "
            f"{widest.total:,.0f} over {widest.span_hours:.0f} hours")

    if not case.flows and not case.fans:
        lines.append("\nNo traced structure - the model flagged these on "
                     "transaction features alone.")

    return "\n".join(lines)


def _one_review(description: str, key: str, model: str,
                temperature: float, timeout: int = 60,
                attempts: int = 4) -> Review:
    """A single panel member. Failures are recorded, never silently dropped.

    Retries on rate limiting. A panel of three across four concurrent cases is
    twelve simultaneous requests, which Groq throttles: a first run without this
    lost **109 of 120 calls**, and every failure became an "unclear" vote. That
    is the dangerous shape of this bug - the panel reported unanimous agreement
    at 100% when in fact nothing had been asked, and the queue looked triaged.
    """
    import random

    last_error = "no attempt made"
    for attempt in range(attempts):
        try:
            response = requests.post(
                GROQ_URL, timeout=timeout,
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"},
                json={"model": model, "temperature": temperature,
                      "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                   {"role": "user", "content": description}]})
            if response.status_code == 200:
                break
            last_error = f"HTTP {response.status_code}"
            if response.status_code in (429, 500, 502, 503) and attempt < attempts - 1:
                # Honour Retry-After when given; jitter otherwise so a whole
                # panel does not retry in lockstep and throttle itself again.
                wait = float(response.headers.get("retry-after", 0)) or                     (2 ** attempt) * (0.5 + random.random())
                time.sleep(min(wait, 30))
                continue
            return Review("unclear", 0.0, "", last_error)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < attempts - 1:
                time.sleep((2 ** attempt) * (0.5 + random.random()))
                continue
            return Review("unclear", 0.0, "", last_error)
    else:
        return Review("unclear", 0.0, "", last_error)

    try:

        text = response.json()["choices"][0]["message"]["content"]
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return Review("unclear", 0.0, text[:120], "no JSON in reply")

        parsed = json.loads(match.group())
        verdict = str(parsed.get("verdict", "unclear")).lower()
        if verdict not in VERDICTS:
            verdict = "unclear"
        # The model returns unicode punctuation (non-breaking hyphens, curly
        # quotes) and a Windows console is cp1252, so printing raw output raises
        # UnicodeEncodeError. Normalise here rather than at every print site.
        reason = str(parsed.get("reason", ""))[:200]
        reason = reason.encode("ascii", "replace").decode("ascii")
        return Review(verdict, float(parsed.get("confidence", 0.0)), reason)
    except Exception as exc:
        return Review("unclear", 0.0, "", f"{type(exc).__name__}: {exc}")


def triage_case(case, key: str | None = None, model: str = DEFAULT_MODEL,
                panel_size: int = PANEL_SIZE,
                temperature: float = TEMPERATURE) -> TriageResult:
    """Run the panel over one case and take the majority verdict."""
    key = key or load_key()
    description = describe_case(case)

    with ThreadPoolExecutor(max_workers=panel_size) as pool:
        reviews = list(pool.map(
            lambda _: _one_review(description, key, model, temperature),
            range(panel_size)))

    # A failed call is not an opinion. Counting one as "unclear" made a panel
    # of three failures look like unanimous agreement, which is worse than
    # returning nothing.
    valid = [r for r in reviews if r.error is None]
    if not valid:
        return TriageResult(case_id=case.case_id, verdict="unreviewed",
                            agreement=0.0, reviews=reviews, escalate=True,
                            summary="panel failed - escalated by default")

    counts: dict[str, int] = {}
    for review in valid:
        counts[review.verdict] = counts.get(review.verdict, 0) + 1

    verdict = max(counts, key=counts.get)
    agreement = counts[verdict] / len(valid)

    # Escalate on genuine division, and on a laundering call regardless. A split
    # panel means the case is ambiguous, which is precisely what a human is for.
    escalate = agreement < 0.67 or verdict == "laundering"

    best = max(valid, key=lambda r: r.confidence)
    return TriageResult(case_id=case.case_id, verdict=verdict,
                        agreement=agreement, reviews=reviews,
                        escalate=escalate, summary=best.reason)


def triage_queue(cases, limit: int = 50, key: str | None = None,
                 model: str = DEFAULT_MODEL, workers: int = 2) -> list[TriageResult]:
    """Triage the top of the queue.

    Only the top: an investigator works downward, and the panel costs a call per
    case per panel member. Triaging 3,895 cases to reorder ones nobody will open
    is money spent on nothing.
    """
    key = key or load_key()
    selected = cases[:limit]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda c: triage_case(c, key, model), selected))


def format_triage(results: list[TriageResult]) -> str:
    lines = [f"{'case':>6} {'verdict':>12} {'agree':>7} {'escalate':>9}  summary",
             "-" * 92]
    for r in results:
        lines.append(f"{r.case_id:>6} {r.verdict:>12} {r.agreement:>6.0%} "
                     f"{'YES' if r.escalate else 'no':>9}  {r.summary[:52]}")
    return "\n".join(lines)
