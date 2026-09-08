/**
 * The loans behind one governed eligibility status.
 *
 * Follows the existing drill-through convention on this tab: the service
 * returns the rows AND the columns, and the UI renders exactly those. It does
 * not choose fields, does not widen the disclosure, and does not reconstruct
 * the population — the same governed determination that produced the eligible
 * balance produces this list, so the two always agree.
 */

import { useEffect, useState } from "react";
import { X } from "lucide-react";
import type { AgentClient } from "@/api/AgentClient";
import type { EligibilityLoans } from "@/domain";
import { Card } from "@/components/ui";

const STATUS_TITLE: Record<string, string> = {
  ELIGIBLE: "Eligible loans",
  INELIGIBLE: "Ineligible loans",
  UNDETERMINED: "Undetermined loans",
};

export function EligibilityLoansPanel({
  client,
  portfolioId,
  portfolioContext,
  status,
  onClose,
}: {
  client: AgentClient;
  portfolioId: string;
  portfolioContext?: string;
  status: "ELIGIBLE" | "INELIGIBLE" | "UNDETERMINED";
  onClose: () => void;
}) {
  const [loans, setLoans] = useState<EligibilityLoans | null>(null);
  const [loading, setLoading] = useState(false);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoans(null);
    setFailed(false);
    setLoading(true);
    const fetchLoans = client.getEligibilityLoans?.bind(client);
    if (!fetchLoans) {
      setLoans({
        available: false,
        reason: "This client does not serve the eligibility drill-down.",
        columns: [],
        rows: [],
      });
      setLoading(false);
      return;
    }
    fetchLoans(portfolioId, status, portfolioContext)
      .then((d) => {
        if (!cancelled) setLoans(d);
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [client, portfolioId, portfolioContext, status]);

  return (
    <Card className="space-y-2 p-3" testId="eligibility-loans-panel">
      <div className="flex items-baseline justify-between gap-2">
        <h3 className="text-[12px] font-semibold text-ink-100">
          {STATUS_TITLE[status] ?? status}
          {loans?.available && (
            <span className="ml-2 font-normal text-[11px] text-ink-500">
              {loans.rowCount?.toLocaleString("en-GB")} loan
              {loans.rowCount === 1 ? "" : "s"}
              {loans.truncated ? ` · showing first ${loans.rows.length}` : ""}
            </span>
          )}
        </h3>
        <button
          type="button"
          aria-label="Close"
          onClick={onClose}
          className="rounded p-0.5 text-ink-500 transition-colors hover:text-ink-100"
        >
          <X size={13} />
        </button>
      </div>

      {loading && !loans && (
        <p className="text-[12px] text-ink-500">Loading loans…</p>
      )}
      {failed && (
        <p className="text-[12px] text-amber-300/90">
          The eligibility drill-down could not be reached.
        </p>
      )}
      {loans && !loans.available && (
        <p className="text-[12px] text-ink-500">{loans.reason}</p>
      )}
      {loans?.available && loans.rows.length === 0 && (
        <p className="text-[12px] text-ink-500">
          No loans currently carry this status.
        </p>
      )}
      {loans?.available && loans.rows.length > 0 && (
        <div className="max-h-64 overflow-auto rounded-lg border border-[var(--color-line-soft)]">
          <table className="w-full text-[11px]">
            <thead>
              <tr className="border-b border-[var(--color-line-soft)] text-left text-[10px] uppercase tracking-wider text-ink-500">
                {loans.columns.map((c) => (
                  <th key={c} className="px-2 py-1 font-medium">
                    {c.replace(/_/g, " ")}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {loans.rows.map((row, i) => (
                <tr key={i} className="border-b border-[var(--color-line-soft)] last:border-0">
                  {loans.columns.map((c) => (
                    <td key={c} className="px-2 py-1 font-mono tabular-nums text-ink-300">
                      {typeof row[c] === "number"
                        ? (row[c] as number).toLocaleString("en-GB")
                        : String(row[c] ?? "—")}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
