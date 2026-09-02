/**
 * Funded → Cohorts.
 *
 * This surface exists to answer a different question from the Evolution tab,
 * and the difference is the whole point:
 *
 *   Evolution   the book outstanding at each reporting date — 33, then 73.
 *   Cohorts     what ENTERED in each origination period — 33, then 40 — and
 *               how one of those vintages behaves as it seasons.
 *
 * It previously rendered the evolution component, so the Cohorts tab showed
 * cumulative portfolio totals under cohort labels. Nothing here shows a
 * running book total; that view is deliberately left where it already lives.
 *
 * Three blocks, in the order an analyst reads them: formation (what entered),
 * the static pool for a selected vintage (how it seasons and who leaves), and
 * point-in-time composition of that vintage.
 */

import { useEffect, useMemo, useState } from "react";
import type { AgentClient } from "@/api";
import type { CohortFormation, CohortStaticPool } from "@/domain";
import { formatGBP, formatValue } from "@/lib/utils";

/** Percent metrics are FRACTIONS by contract (0.395 == 39.5%). */
function pct(v: number | null | undefined): string {
  return v == null ? "—" : formatValue(v, "pct", "percent_fraction");
}
function gbp(v: number | null | undefined): string {
  return v == null ? "—" : formatGBP(v);
}

export function CohortsPanel({ client, portfolioId, portfolioContext }: {
  client: AgentClient;
  portfolioId: string;
  portfolioContext?: string;
}) {
  const [formation, setFormation] = useState<CohortFormation | null>(null);
  const [pool, setPool] = useState<CohortStaticPool | null>(null);
  const [vintage, setVintage] = useState<string>("");
  const [grain, setGrain] = useState<"M" | "Q" | "Y">("M");

  useEffect(() => {
    let live = true;
    client.getCohortVintages(portfolioId, { portfolioContext, grain })
      .then((r) => { if (live && r.dataset === "cohort_formation") setFormation(r); })
      .catch(() => { if (live) setFormation(null); });
    return () => { live = false; };
  }, [client, portfolioId, portfolioContext, grain]);

  const vintages = useMemo(() => formation?.vintages ?? [], [formation]);
  // Share of the book AT ENTRY. Summed from the balances the engine returned;
  // it is a presentation ratio over one table, not a new economic measure.
  const totalEntryBalance = useMemo(
    () => vintages.reduce((sum, v) => sum + (v.originalBalance ?? 0), 0),
    [vintages],
  );
  // Default to the OLDEST vintage: it has the most seasoning to show.
  const selected = vintage || vintages[0]?.vintage || "";

  useEffect(() => {
    if (!selected) { setPool(null); return; }
    let live = true;
    client.getCohortVintages(portfolioId, { portfolioContext, grain, vintage: selected })
      .then((r) => { if (live && r.dataset === "cohort_static_pool") setPool(r); })
      .catch(() => { if (live) setPool(null); });
    return () => { live = false; };
  }, [client, portfolioId, portfolioContext, grain, selected]);

  return (
    <div className="space-y-3" data-testid="cohorts-panel">
      <div className="flex flex-wrap items-end gap-3 rounded-xl border border-[var(--color-line)] bg-navy-900/40 px-3 py-2.5">
        <label className="flex flex-col gap-0.5">
          <span className="text-[10px] uppercase tracking-wide text-ink-500">Vintage grain</span>
          <select className="rounded-md border border-[var(--color-line)] bg-navy-950 px-2 py-1 text-[11px] text-ink-100"
            value={grain} onChange={(e) => { setGrain(e.target.value as "M" | "Q" | "Y"); setVintage(""); }}
            data-testid="vintage-grain">
            <option value="M">Month</option><option value="Q">Quarter</option><option value="Y">Year</option>
          </select>
        </label>
        <p className="self-end pb-1 text-[10px] text-ink-500">
          Cohort entry is the loan&rsquo;s origination (policy completion) date, so a loan
          belongs to one vintage for life. The book outstanding at each reporting date is
          on the Evolution tab.
        </p>
      </div>

      {/* A. Formation — what entered, each loan counted once. */}
      <div className="text-[11px] font-semibold text-ink-300">
        Vintage formation — loans entering the book in each origination period
      </div>
      {formation && !formation.available ? (
        <div className="rounded-lg border border-amber-400/20 bg-amber-400/5 px-3 py-2 text-[11px] text-amber-300/90"
          data-testid="formation-unavailable">
          No vintage formation{formation.reason ? ` — ${formation.reason}` : ""}.
        </div>
      ) : (
        <div className="overflow-x-auto rounded-xl border border-[var(--color-line)] bg-navy-900/40">
          <table className="w-full text-[11px]" data-testid="formation-table">
            <thead>
              <tr className="border-b border-[var(--color-line)] text-ink-400">
                <th className="px-3 py-2 text-left font-medium">Vintage</th>
                <th className="px-3 py-2 text-right font-medium">Loans entering</th>
                <th className="px-3 py-2 text-right font-medium">Balance at entry</th>
                <th className="px-3 py-2 text-right font-medium">Share of book</th>
                <th className="px-3 py-2 text-right font-medium"
                  title="Weighted-average LTV at origination. Only vintages whose tape supplies an original LTV can report one — an acquired book usually cannot.">
                  WA original LTV</th>
                <th className="px-3 py-2 text-right font-medium"
                  title="Weighted-average CURRENT LTV of the loans as they entered the book. Available for every vintage, including acquired ones with no origination LTV.">
                  WA LTV at entry</th>
                <th className="px-3 py-2 text-right font-medium">WA rate</th>
              </tr>
            </thead>
            <tbody>
              {vintages.map((v) => (
                <tr key={v.vintage}
                  className={"cursor-pointer border-b border-[var(--color-line-soft)] last:border-0 "
                    + (v.vintage === selected ? "bg-peri-400/10" : "hover:bg-navy-800/40")}
                  onClick={() => setVintage(v.vintage)} data-testid={`vintage-${v.vintage}`}>
                  <td className="px-3 py-1.5 text-left font-medium text-ink-100">{v.vintage}</td>
                  <td className="px-3 py-1.5 text-right text-ink-200">{v.originalLoanCount.toLocaleString("en-GB")}</td>
                  <td className="px-3 py-1.5 text-right text-ink-200">{gbp(v.originalBalance)}</td>
                  <td className="px-3 py-1.5 text-right text-ink-200">
                    {totalEntryBalance
                      ? `${((v.originalBalance / totalEntryBalance) * 100).toFixed(1)}%`
                      : "—"}</td>
                  <td className="px-3 py-1.5 text-right text-ink-200">{pct(v.waOriginalLtv)}</td>
                  <td className="px-3 py-1.5 text-right text-ink-200">{pct(v.waEntryLtv)}</td>
                  <td className="px-3 py-1.5 text-right text-ink-200">{pct(v.waRate)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {(formation?.lateCorrections?.length ?? 0) > 0 && (
        <div className="rounded-lg border border-amber-400/20 bg-amber-400/5 px-3 py-2 text-[11px] text-amber-300/90"
          data-testid="late-corrections">
          {formation!.lateCorrections!.length} loan(s) were restated into a different
          origination period by a later snapshot. Each keeps the vintage it was first
          assigned, so the pools below stay fixed.
        </div>
      )}

      {/* B. Static pool — one vintage followed forward. */}
      <div className="pt-1 text-[11px] font-semibold text-ink-300">
        Static pool — {selected || "no vintage"} followed through reporting periods
      </div>
      {pool && !pool.available ? (
        <div className="rounded-lg border border-amber-400/20 bg-amber-400/5 px-3 py-2 text-[11px] text-amber-300/90"
          data-testid="pool-unavailable">
          No static pool{pool.reason ? ` — ${pool.reason}` : ""}.
        </div>
      ) : (
        <div className="overflow-x-auto rounded-xl border border-[var(--color-line)] bg-navy-900/40">
          <table className="w-full text-[11px]" data-testid="static-pool-table">
            <thead>
              <tr className="border-b border-[var(--color-line)] text-ink-400">
                <th className="px-3 py-2 text-left font-medium">Period</th>
                <th className="px-3 py-2 text-right font-medium">Seasoning</th>
                <th className="px-3 py-2 text-right font-medium">Surviving</th>
                <th className="px-3 py-2 text-right font-medium">Balance</th>
                <th className="px-3 py-2 text-right font-medium">Balance retention</th>
                <th className="px-3 py-2 text-right font-medium">Exits</th>
                <th className="px-3 py-2 text-right font-medium">WA LTV</th>
              </tr>
            </thead>
            <tbody>
              {(pool?.periods ?? []).map((p) => (
                <tr key={p.period}
                  className="border-b border-[var(--color-line-soft)] last:border-0">
                  <td className="px-3 py-1.5 text-left font-medium text-ink-100">
                    {p.period}
                    {p.forming && (
                      <span
                        title="The vintage was still originating at this date, so the pool was not yet fixed and there is no survival rate to report."
                        className="ml-2 rounded-full border border-peri-400/30 bg-peri-400/10 px-1.5 py-0.5 text-[9px] font-medium uppercase tracking-wider text-peri-200"
                      >Forming</span>
                    )}
                  </td>
                  <td className="px-3 py-1.5 text-right text-ink-400">
                    {p.monthsSinceEntry == null ? "—" : `+${p.monthsSinceEntry} mo`}</td>
                  <td className="px-3 py-1.5 text-right text-ink-200">{p.survivingLoanCount.toLocaleString("en-GB")}</td>
                  <td className="px-3 py-1.5 text-right text-ink-200">{gbp(p.currentBalance)}</td>
                  <td className="px-3 py-1.5 text-right text-ink-200">{pct(p.balanceRetention)}</td>
                  <td className="px-3 py-1.5 text-right text-ink-200">
                    {p.exitsInPeriod > 0 ? `${p.exitsInPeriod} (${p.cumulativeExits} cum.)` : "—"}</td>
                  <td className="px-3 py-1.5 text-right text-ink-200">{pct(p.waLtv)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="text-[10px] text-ink-500">
        The pool is fixed once the vintage stops originating{pool?.formationEnd
          ? ` (${pool.formationEnd})` : ""}. From that point a falling count is redemption
        or exit and the count can never rise; balance retention above 100% is interest
        roll-up, not new lending. Periods marked <span className="text-peri-200">Forming</span>{" "}
        pre-date that: the vintage was still admitting loans, so no retention is shown.
      </p>
    </div>
  );
}
