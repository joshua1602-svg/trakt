/**
 * Renders the REAL Eligibility & Concentrations workspace against a REAL
 * governed envelope.
 *
 * The JSON beside this file is the wire payload
 * `mi_agent_api.concentration_tests_api.compute_concentration_tests` produced
 * for a synthetic borrower base — same frames, same evaluation, same service
 * the live dashboard calls. Nothing is drawn here that the product does not
 * draw; this file only stands in for the HTTP client so a screenshot does not
 * need a running API.
 *
 *     npx vite --port 5199
 *     open http://localhost:5199/preview/borrowing-base.html?state=governed
 */

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import type { AgentClient } from "@/api/AgentClient";
import type { ConcentrationTestsSnapshot, EligibilityLoans } from "@/domain";
import { RiskLimitsWorkspace } from "@/components/risk/RiskLimitsWorkspace";
import governed from "./data/governed.json";
import prototype from "./data/prototype.json";
import "../src/index.css";

const STATES: Record<string, unknown> = { governed, prototype };
const state = new URLSearchParams(location.search).get("state") ?? "governed";
const snapshot = (STATES[state] ?? governed) as ConcentrationTestsSnapshot;

/** The loans behind one eligibility status, derived from the same envelope. */
function eligibilityLoans(status: string): EligibilityLoans {
  const base = snapshot.borrowingBase;
  const counts: Record<string, number> = {
    ELIGIBLE: base?.eligibleLoanCount ?? 0,
    INELIGIBLE: base?.ineligibleLoanCount ?? 0,
    UNDETERMINED: base?.undeterminedLoanCount ?? 0,
  };
  const reason =
    status === "ELIGIBLE"
      ? "all_approved_eligibility_rules_satisfied"
      : status === "INELIGIBLE"
        ? "more_than_30_days_past_due"
        : "eligibility_rule_input_missing:no_material_arrears";
  const rowCount = counts[status] ?? 0;
  return {
    available: true,
    status: status as EligibilityLoans["status"],
    facilityId: base?.facilityId ?? null,
    columns: [
      "loan_id",
      "collateral_geography",
      "current_outstanding_balance",
      "original_valuation_amount",
      "borrowing_base_eligibility_status",
      "borrowing_base_eligibility_reason",
    ],
    rows: Array.from({ length: Math.min(rowCount, 12) }, (_, i) => ({
      loan_id: `ERE${100_000 + i * 7}`,
      collateral_geography: ["East Of England", "London", "South East"][i % 3],
      current_outstanding_balance: 118_400 + i * 9_310,
      original_valuation_amount: 402_000 + i * 21_500,
      borrowing_base_eligibility_status: status,
      borrowing_base_eligibility_reason: reason,
    })),
    rowCount,
    truncated: rowCount > 12,
    reportingDate: snapshot.reportingDate,
    toRunId: snapshot.toRunId,
  };
}

const client = {
  id: "preview",
  mock: true,
  getConcentrationTests: async () => snapshot,
  getEligibilityLoans: async (_p: string, status: string) =>
    eligibilityLoans(status),
  getConcentrationDrillthrough: async () => ({
    available: false,
    reason: "Drill-through is served by the live API.",
    columns: [],
    rows: [],
  }),
  getConcentrationHistory: async () => ({ available: false, series: [] }),
  getConcentrationDrivers: async () => ({ available: false, drivers: [] }),
} as unknown as AgentClient;

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <div className="min-h-screen bg-[var(--color-bg)] p-6">
      <div className="mx-auto max-w-[1180px]">
        <RiskLimitsWorkspace client={client} portfolioId="ere_funding_uk" />
      </div>
    </div>
  </StrictMode>,
);
