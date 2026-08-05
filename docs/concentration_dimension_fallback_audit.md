# Concentration dimension fallback — active-configuration audit

**Decision: RETAIN the P6 risk-library changes, including the dimension
fallback. No committed or live-style configuration changes behaviour.**

Machine-readable companion: [`concentration_dimension_fallback_audit.csv`](concentration_dimension_fallback_audit.csv)
(regenerate with the snippet in §5).

---

## 1. What changed and why it needed auditing

`mi_agent/concentration_tests/metrics.py` previously resolved a governed
`dimension` parameter through a fixed table only:

```python
role = _DIMENSION_ROLES.get(dimension)     # 14 entries, or None
```

It now falls back to a role of the SAME NAME when one is declared in the
library:

```python
def _dimension_role(lib, dimension):
    role = _DIMENSION_ROLES.get(name)
    if role:
        return role
    return name if (name and lib.role(name) is not None) else None
```

That is what lets a new composition dimension (`manufacturer`, `vendor`,
`industry`, `charge_type`, …) be added as configuration rather than as code.

**The risk it introduces:** a dimension name that previously resolved to
`None` — and therefore produced an `unavailable` test — may now resolve and
**measure**. A measured test has a RAG state. A client's risk position could
move from "unavailable" to "pass/warning/breach" with no configuration change
and no operator approval.

`config/risk/concentration_test_library.yaml` declares **50 field roles**, of
which **36** are not in the 14-entry table and are therefore newly resolvable:

```
account_status, arrears_balance, balance_current, balance_original,
balloon_amount, borrower_age_oldest, borrower_age_youngest, borrower_count,
borrower_id, borrower_structure, charge_type, collateral_type, contract_type,
country, county, days_past_due, default_date, industry, interest_rate,
interest_rate_margin, lien, loan_id, ltv_current, ltv_indexed, ltv_original,
manufacturer, maturity_date, original_term, origination_date, postcode,
residual_value_current, residual_value_original, valuation_current,
valuation_indexed, valuation_original, vendor
```

The 14 pre-existing dimensions (`broker`, `loan_purpose`, `originator`,
`payment_option`, `product`, `property_type`, `rate_type`, `region`, `seller`,
`servicer`, `source_portfolio`, `source_type`, `spv`, `tenure`) are consulted
from the explicit table **first**, so their behaviour is bit-for-bit unchanged.

## 2. Method

Every `.yaml` / `.yml` / `.json` file in the repository (excluding
`node_modules`, `demo-video`, `frontend`, `landing-page`) was parsed and walked
for any object carrying a `dimension` key whose value is one of the 36 names.
Each hit was then resolved through the **production** conversion path,
`mi_agent.concentration_tests.compat.proposals_from_extracted_config`, to
establish what it actually becomes at runtime.

## 3. Result — one hit, no behaviour change

| Field | Value |
|---|---|
| Configuration file | `config/clients/client_001/risk_limits_extracted.yaml` |
| Test / limit id | `joint_borrower_limit_borrower_structure` |
| Dimension | `borrower_structure` |
| Category | `joint_borrower_limit` |
| `limit_value` | **`None`** — no threshold |
| `needs_review` | **`True`** |
| Previous state | dimension unresolvable → metric unavailable |
| New state | dimension resolvable as a same-named field role |
| Resolved status | **`pending_confirmation`** |
| Match outcome | **`ambiguous`** |
| Resolved parameters | **`{}`** |
| Could the metric or RAG status change? | **No** |

**Three independent reasons this cannot change behaviour:**

1. **The `dimension` key is never read.** `compat.py` builds metric parameters
   from `category`, `limit_value`, `direction`, `source_snippet`, `limit_id`,
   `needs_review` and `region`. It never reads `lim["dimension"]`. The only
   `dimension` parameter it ever sets is `{"dimension": "broker"}` for broker
   concentration — and `broker` is in the pre-existing table, unaffected by the
   fallback. Verified: resolved parameters are `{}`.
2. **It is a proposal, not an active configuration.** The file is *extracted*
   limits (`status: needs_review`), consumed as a **fallback** by
   `mi_agent_api/risk_limits.py` and converted by `compat.py` into
   `TestProposal` objects with `status: pending_confirmation`. It cannot become
   an `ActiveTest` without explicit operator approval.
3. **There is no threshold.** `limit_value` is `None`, so `compat.py` forces
   `MATCH_AMBIGUOUS` (`THRESHOLD_UNIT_UNCERTAIN`), and `needs_review: True`
   forces it again. Even if the dimension resolved and reached the metric,
   there is nothing to compare a measurement against, so no RAG state exists.

**No `ActiveConfiguration` is committed anywhere in the repository.** Active
configurations are minted by the operator approval workflow at runtime and
stored, not checked in — so the audit above covers everything that exists to
be audited in-tree.

## 4. Decision

**Retain the P6 changes.** The condition the reviewer set — *"remove/narrow the
fallback if a live configuration could move from unavailable to measured
without explicit approval"* — is not met by any configuration present.

Residual risk, stated rather than dismissed: a **runtime** `ActiveConfiguration`
that this repository does not contain could, in principle, name one of the 36
role names as a `dimension`. Such a configuration would have had to be approved
against a test that was returning `unavailable`, which is not a state an
operator would knowingly approve. It is worth confirming against a production
store before release; it is not something the repository can answer.

## 5. Regenerating this audit

```bash
python - <<'PY'
import sys, json, yaml, pathlib, dataclasses
sys.path.insert(0, '.')
from mi_agent.concentration_tests.metrics import _DIMENSION_ROLES
from mi_agent.concentration_tests import compat
lib = yaml.safe_load(open('config/risk/concentration_test_library.yaml'))
newly = [r for r in sorted(lib['field_roles']) if r not in _DIMENSION_ROLES]
print(len(newly), 'newly resolvable role names')
PY
```

The full scan that produced the CSV is pinned as a test in
`tests/test_concentration_dimension_fallback.py`, so a future configuration
that DOES use one of these names will fail the build rather than change a RAG
state quietly.
