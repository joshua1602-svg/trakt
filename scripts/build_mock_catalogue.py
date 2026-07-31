"""Emit the browser fixture's copy of the governed onboarding catalogue.

The catalogue is authored once, in ``config/onboarding/field_catalogue.yaml``,
and served to the real browser over ``/ops/onboarding/reference``. The mock
client — used for demonstration and browser tests — cannot call the API, so it
needs a static copy.

Generating that copy rather than hand-writing it means the fixture can never
quietly disagree with the governed model about what a field is called, who
supplies it, or when it is required.

    python -m scripts.build_mock_catalogue
"""

from __future__ import annotations

import json
from pathlib import Path

from operations_control.onboarding.catalogue import load
from operations_control.onboarding.inference import rules as inference_rules
from operations_control.onboarding.case import KIND_LABELS, STATUS_LABELS
from operations_control.onboarding.service import STEPS, STEP_LABELS

REPO = Path(__file__).resolve().parents[1]
OUT = (REPO / "frontend/operations-control-ui/src/api/mockCatalogue.ts")

HEADER = """/**
 * GENERATED from config/onboarding/field_catalogue.yaml — do not edit by hand.
 *
 * Regenerate with:
 *     python -m scripts.build_mock_catalogue
 *
 * The real browser reads this model from the API. This copy exists only so the
 * mock client can render the same wizard without a backend, and is generated
 * so the fixture cannot drift from the governed catalogue.
 */

import type { CaseKind, CaseStatus, Catalogue } from "./onboardingTypes";

"""


def main() -> None:
    catalogue = load()
    doc = catalogue.to_dict()
    # The fixture needs the shape, not the prose bulk of every regime product.
    body = json.dumps(doc, indent=2, sort_keys=False, ensure_ascii=False)
    steps = json.dumps([{"key": s, "label": STEP_LABELS[s]} for s in STEPS],
                       indent=2, ensure_ascii=False)
    statuses = json.dumps(STATUS_LABELS, indent=2, ensure_ascii=False)
    kinds = json.dumps(KIND_LABELS, indent=2, ensure_ascii=False)
    OUT.write_text(
        HEADER
        + f"export const CATALOGUE = {body} as unknown as Catalogue;\n\n"
        + f"export const STEPS: {{ key: string; label: string }}[] = {steps};\n\n"
        + f"export const STATUS_LABELS: Record<CaseStatus, string> = {statuses};\n\n"
        + f"export const KIND_LABELS: Record<CaseKind, string> = {kinds};\n\n"
        + "export const INFERENCE_RULES = "
        + json.dumps(inference_rules(), indent=2, ensure_ascii=False)
        + " as Record<string, any>;\n",
        encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)} "
          f"({len(doc['sections'])} sections)")


if __name__ == "__main__":
    main()
