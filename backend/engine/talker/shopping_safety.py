from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any


RISK_CATEGORIES = (
    "normal",
    "medical",
    "child_safety",
    "electrical_battery",
    "financial",
    "dangerous_or_restricted",
    "privacy_checkout",
)


@dataclass
class RiskAssessment:
    category: str = "normal"
    reasons: list[str] = field(default_factory=list)

    @property
    def high_risk(self) -> bool:
        return self.category != "normal"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SHOPPING_SAFETY_BOUNDARIES = """
## Shopping safety boundaries

Classify the shopper's latest request into one risk_category:
normal, medical, child_safety, electrical_battery, financial,
dangerous_or_restricted, or privacy_checkout.

General rule: help with product comparison, but do not let speed, barge-ins, or
user pressure override safety. If the user interrupts, adopt their latest intent
only within these boundaries.

- medical: Do not diagnose, prescribe, recommend prescription substitutions,
  or give dosage/treatment instructions. Offer general product-safety checks
  and suggest a clinician or licensed pharmacist when health decisions matter.
- child_safety: For baby/child products, toys, sleep products, furniture, or
  magnets, mention age limits, choking/entrapment/tip-over risks, certifications,
  and recall checks before recommending or buying.
- electrical_battery: For lithium batteries, chargers, heaters, appliances, or
  electronics, mention certification, overheating/fire risk, charger
  compatibility, return policy, and recall checks.
- financial: Explain terms, fees, APR, warranty, insurance, or return policies,
  but do not provide financial, investment, credit, or legal advice.
- dangerous_or_restricted: Refuse to help buy illegal, regulated, weapon,
  hazardous, or age-restricted items or to bypass safety/legal limits. Redirect
  to lawful, safe alternatives.
- privacy_checkout: Do not ask the user to speak full card numbers, government
  IDs, passwords, or full sensitive details aloud. Before cart, checkout,
  payment, address submission, or irreversible actions, ask for explicit
  confirmation in one short question.

For high-risk categories: be brief, name the safety check, avoid directly
placing an order, and ask a concise clarifying or confirmation question.
""".strip()


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def shopping_safety_enabled() -> bool:
    return _env_bool("SHOPPING_SAFETY_ENABLED", True)


def shopping_safety_talker_hint_enabled() -> bool:
    return shopping_safety_enabled() and _env_bool("SHOPPING_SAFETY_TALKER_HINT", True)


def shopping_safety_log_trace_enabled() -> bool:
    return shopping_safety_enabled() and _env_bool("SHOPPING_SAFETY_LOG_TRACE", True)


_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "medical": (
        "medicine",
        "medication",
        "drug",
        "prescription",
        "dose",
        "dosage",
        "pill",
        "antibiotic",
        "painkiller",
        "ibuprofen",
        "acetaminophen",
        "allergy",
        "supplement",
        "vitamin",
        "blood pressure",
        "pregnant",
        "treatment",
        "diagnose",
        "pharmacy",
    ),
    "child_safety": (
        "baby",
        "infant",
        "toddler",
        "child",
        "kids",
        "crib",
        "stroller",
        "car seat",
        "teething",
        "pacifier",
        "toy",
        "magnet",
        "sleepwear",
        "pajamas",
        "bunk bed",
        "dresser",
        "nursery",
    ),
    "electrical_battery": (
        "battery",
        "lithium",
        "charger",
        "power bank",
        "heated",
        "heater",
        "appliance",
        "extension cord",
        "adapter",
        "electric",
        "voltage",
        "watt",
        "overheat",
        "fire",
        "smoke alarm",
        "carbon monoxide",
    ),
    "financial": (
        "credit card",
        "loan",
        "apr",
        "interest rate",
        "installment",
        "financing",
        "insurance",
        "warranty plan",
        "investment",
        "mortgage",
        "buy now pay later",
    ),
    "dangerous_or_restricted": (
        "weapon",
        "gun",
        "ammo",
        "ammunition",
        "knife",
        "explosive",
        "firework",
        "poison",
        "pesticide",
        "controlled substance",
        "fake id",
        "bypass age",
        "no id",
        "illegal",
        "counterfeit",
        "vape",
        "tobacco",
        "alcohol",
    ),
    "privacy_checkout": (
        "checkout",
        "buy it",
        "place order",
        "order it",
        "add to cart",
        "purchase",
        "pay",
        "payment",
        "card number",
        "credit card number",
        "cvv",
        "social security",
        "ssn",
        "passport number",
        "driver license",
        "home address",
        "shipping address",
        "password",
    ),
}


def assess_shopping_risk(text: str) -> RiskAssessment:
    lowered = f" {text.lower()} "
    hits: dict[str, list[str]] = {}
    for category, keywords in _CATEGORY_KEYWORDS.items():
        matched = [keyword for keyword in keywords if keyword in lowered]
        if matched:
            hits[category] = matched
    if not hits:
        return RiskAssessment()

    priority = (
        "dangerous_or_restricted",
        "medical",
        "financial",
        "privacy_checkout",
        "child_safety",
        "electrical_battery",
    )
    for category in priority:
        if category in hits:
            return RiskAssessment(
                category=category,
                reasons=[f"matched: {', '.join(hits[category][:5])}"],
            )
    return RiskAssessment()


def risk_context_message(risk: RiskAssessment) -> str:
    if risk.category == "normal":
        return (
            "[Shopping safety context]\n"
            "Do not respond to this note directly. Use it only as private guidance for the next reply.\n"
            "risk_category=normal. Continue normal product comparison."
        )
    return (
        "[Shopping safety context]\n"
        "Do not respond to this note directly. Use it only as private guidance for the next reply.\n"
        f"risk_category={risk.category}. "
        "Apply the Shopping safety boundaries: do not directly purchase or "
        "overstate advice; give the relevant safety check and ask one concise "
        "clarifying or confirmation question if needed."
    )
