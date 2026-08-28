"""Context for a support agent when a customer asks why a card was declined.

**This does not explain the model's decision, and saying that it does would be
false.** Measured on the 200 highest-scoring card transactions, only **25.3% of
the evidence that pushed those scores up sits on features whose meaning is
known** - median 18.7%, and in 82% of cases under a quarter. The rest is
`V258`, `C1`, `V187` and their neighbours, which IEEE-CIS anonymised
deliberately and which have no human meaning to recover.

An explanation built on a quarter of the reasoning would sound authoritative and
be mostly wrong, and an agent repeating it would tell a customer something
false. That is worse than saying nothing.

So this answers a different, answerable question: **what is factually unusual
about this transaction, compared with what this card normally does?** Every line
it produces is a measured comparison against that card's own history. None of it
is a claim about why the model scored what it did.

The distinction matters legally as well as technically. A bank may tell a
customer their payment looked unusual; inventing a specific false reason is a
different thing entirely.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"

# Low, because this writes something an agent reads to a customer. Variety is
# not a virtue here - the same facts should produce the same summary.
TEMPERATURE = 0.2

SYSTEM_PROMPT = """You write one short paragraph for a bank's support agent whose
customer is asking why their card payment was declined.

You will be given a list of measured facts about the transaction. Write two or
three sentences the agent can read and understand.

Rules:
- Use ONLY the facts given. Invent nothing. If a fact is not listed, it is not known.
- Do not say why the system declined it. You do not know that. Say what was
  unusual about the transaction.
- Plain language. The agent is not technical.
- No apology, no hedging, no speculation about fraud.

If the facts are thin, say so plainly - "little of this transaction stood out on
its own" is a useful and honest thing for an agent to know."""


@dataclass
class Fact:
    """One measured comparison. Both halves are values, never inferences."""

    label: str
    detail: str

    def __str__(self) -> str:
        return f"{self.label}: {self.detail}"


# The model returns unicode punctuation - narrow no-break spaces around numbers,
# curly quotes, en dashes - and this text reaches a Windows console, a CSV and
# eventually an agent's screen. Encoding with errors="replace" turns each into a
# literal "?" mid-sentence, which reads as corruption; mapping them to their
# ASCII equivalents keeps the sentence intact.
_ASCII = {
    " ": " ", " ": " ", " ": " ", " ": " ",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": " - ", "…": "...",
}


def _to_ascii(text: str) -> str:
    for bad, good in _ASCII.items():
        text = text.replace(bad, good)
    return text.encode("ascii", "ignore").decode("ascii")


def _money(value: float) -> str:
    return f"{value:,.0f}"


def build_facts(features: pd.Series) -> list[Fact]:
    """Compare this transaction with the card's own history.

    Reads only columns whose meaning is documented. Anything absent is skipped
    rather than guessed at - IEEE-CIS is heavily missing, and a fact asserted
    from a NaN is a fabrication.
    """
    facts: list[Fact] = []

    def get(name):
        value = features.get(name)
        return None if value is None or (isinstance(value, float) and np.isnan(value)) else value

    amount = get("TransactionAmt")
    if amount is not None:
        facts.append(Fact("Amount", _money(amount)))

    # The card's own baseline. This is the comparison an agent can actually use.
    card_mean = get("TransactionAmt_uid_mean")
    if amount is not None and card_mean:
        ratio = amount / card_mean
        if ratio >= 2 or ratio <= 0.5:
            facts.append(Fact(
                "Against this card's usual spend",
                f"typically {_money(card_mean)}, so this is "
                f"{ratio:.1f} times {'higher' if ratio > 1 else 'lower'}"))
        else:
            facts.append(Fact("Against this card's usual spend",
                              f"typically {_money(card_mean)} - broadly in line"))

    seen = get("uid_count")
    if seen is not None:
        facts.append(Fact(
            "History for this card",
            f"{int(seen)} previous transactions on record" if seen > 1
            else "no previous transactions on record - this card is new to us"))

    hour = get("hour")
    if hour is not None:
        h = int(hour)
        if h <= 5:
            facts.append(Fact("Time of day", f"{h:02d}:00 - overnight"))
        else:
            facts.append(Fact("Time of day", f"{h:02d}:00"))

    device = get("DeviceType")
    device_info = get("DeviceInfo")
    if device is not None or device_info is not None:
        facts.append(Fact("Device", "recorded for this payment"))
    else:
        facts.append(Fact("Device", "no device details came through with this payment"))

    distance = get("dist1")
    if distance is not None:
        facts.append(Fact("Distance from the billing address", f"{distance:,.0f}"))

    country = get("addr2")
    if country is not None:
        facts.append(Fact("Billing country code", str(int(country))))
    else:
        facts.append(Fact("Billing country", "not supplied with this payment"))

    return facts


def summarise(facts: list[Fact], key: str | None = None,
              model: str = DEFAULT_MODEL, timeout: int = 45) -> dict:
    """Turn the facts into a paragraph an agent can read aloud.

    Returns the summary *and* the facts it was built from, so the agent can see
    exactly what it rests on. The check below then confirms the model did not
    add a number of its own.
    """
    key = key or _load_key()
    listing = "\n".join(f"- {f}" for f in facts)

    try:
        response = requests.post(
            GROQ_URL, timeout=timeout,
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            json={"model": model, "temperature": TEMPERATURE,
                  "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                               {"role": "user", "content": listing}]})
        if response.status_code != 200:
            return {"summary": "", "facts": [str(f) for f in facts],
                    "error": f"HTTP {response.status_code}"}
        text = response.json()["choices"][0]["message"]["content"].strip()
        text = _to_ascii(text)
    except Exception as exc:
        return {"summary": "", "facts": [str(f) for f in facts],
                "error": f"{type(exc).__name__}: {exc}"}

    grounded, invented = verify(text, facts)
    return {"summary": text, "facts": [str(f) for f in facts],
            "grounded": grounded, "invented_numbers": invented,
            "error": None}


def verify(summary: str, facts: list[Fact]) -> tuple[bool, list[str]]:
    """Did the model stick to the facts it was given?

    Checks every number in the summary against the numbers in the facts. An
    agent repeating a fabricated figure to a customer is the failure this
    feature exists to avoid, so it is checked rather than trusted.
    """
    def numbers(text: str) -> set[str]:
        return {n.replace(",", "").rstrip(".0") or "0"
                for n in re.findall(r"\d[\d,]*\.?\d*", text)}

    given = set()
    for fact in facts:
        given |= numbers(str(fact))
    # Multiples and percentages the model may reasonably derive.
    given |= {str(i) for i in range(0, 25)}

    invented = sorted(numbers(summary) - given)
    return not invented, invented


def _load_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key
    env = Path(__file__).resolve().parents[2] / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("GROQ_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("GROQ_API_KEY not set and not found in .env")


def explain_for_agent(features: pd.Series, action: str,
                      key: str | None = None) -> dict:
    """The whole thing: facts, summary, and what the agent may say.

    `action` is the decision that was taken, because what an agent may tell a
    customer differs by outcome - a step-up is a request, a block is a refusal.
    """
    facts = build_facts(features)
    result = summarise(facts, key=key)

    result["action"] = action
    result["disclaimer"] = (
        "Context only. This describes what was unusual about the transaction, "
        "not why the system declined it - most of the model's evidence sits in "
        "anonymised fields with no readable meaning.")
    return result
