from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_GUARDRAILS_VENDOR_PATH = r"C:\gr_pkgs"


@dataclass(slots=True)
class GuardrailsVendorLoadResult:
    available: bool
    reason: str
    restrict_to_topic_class: type | None = None


def load_guardrails_vendor() -> GuardrailsVendorLoadResult:
    vendor_path = Path(
        os.getenv("GUARDRAILS_VENDOR_PATH", DEFAULT_GUARDRAILS_VENDOR_PATH)
    )
    if not vendor_path.exists():
        return GuardrailsVendorLoadResult(
            available=False,
            reason=(
                f"Guardrails vendor path not found: {vendor_path}. "
                "Run scripts/install_guardrails_validator.py first."
            ),
        )

    vendor_path_str = str(vendor_path)
    if vendor_path_str not in sys.path:
        sys.path.insert(0, vendor_path_str)

    try:
        from validator.main import RestrictToTopic
    except Exception as exc:
        return GuardrailsVendorLoadResult(
            available=False,
            reason=f"Failed to import RestrictToTopic from vendor path: {exc}",
        )

    return GuardrailsVendorLoadResult(
        available=True,
        reason=f"Loaded RestrictToTopic from {vendor_path}.",
        restrict_to_topic_class=RestrictToTopic,
    )
