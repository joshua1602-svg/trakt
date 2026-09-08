import { useState } from "react";
import { ChevronDown } from "lucide-react";
import type { Artifact, Reconciliation, SourceNote } from "@/domain";
import { cn, formatGBP } from "@/lib/utils";

/**
 * The reconciliation / coverage footer shown beneath an MI artifact so a result
 * total can always be tied back to the funded-book snapshot. When some balance
 * is excluded (the operator asked to exclude missing dimensions) it says so
 * plainly, and surfaces any field-provenance source notes.
 *
 * SIX FIELDS FOR "NOTHING TO SEE". When coverage is 100%, nothing is excluded
 * and every record is included, the grid spends six lines saying the result
 * reconciles — under every chart, on every answer. That reading is worth ONE
 * line; the detail is a click away for anyone auditing.
 *
 * A footer that has something to report is never collapsed: excluded balance,
 * coverage below 100%, records dropped, applied filters or source notes all
 * open the full grid by default, because those are the cases the footer exists
 * for. Collapsing is only ever applied to the clean case.
 */
function pct(v: number | null | undefined): string {
  return v == null ? "—" : `${v}%`;
}

/** True when the result ties out completely and there is nothing to disclose. */
function isClean(recon: Reconciliation, notes: SourceNote[]): boolean {
  const excluded = recon.balance_excluded_missing ?? 0;
  const included = recon.records_included;
  const total = recon.total_records;
  return (
    notes.length === 0
    && excluded === 0
    && recon.coverage_by_balance_pct === 100
    && !recon.filters_applied
    && included != null && total != null && included === total
  );
}

export function ReconciliationFooter({ artifact }: { artifact: Artifact }) {
  const recon = (artifact as { reconciliation?: Reconciliation }).reconciliation;
  const notes = (artifact as { sourceNotes?: SourceNote[] }).sourceNotes ?? [];
  const clean = !!recon && isClean(recon, notes);
  const [open, setOpen] = useState(false);
  if (!recon && notes.length === 0) return null;

  const excluded = recon?.balance_excluded_missing ?? 0;
  const included = recon?.balance_included ?? null;
  const total = recon?.total_balance ?? null;
  const fields = recon?.missing_dimension_fields ?? [];
  const expanded = !clean || open;

  return (
    <div
      data-testid="reconciliation-footer"
      data-clean={clean ? "true" : "false"}
      className="mt-3 rounded-lg border border-[var(--color-line-soft)] bg-navy-900/50 px-3 py-2 text-[11px] text-ink-400"
    >
      {recon && (
        <>
          {clean ? (
            <button
              type="button"
              onClick={() => setOpen((o) => !o)}
              aria-expanded={expanded}
              className="flex w-full items-center gap-1.5 text-left text-[11px] text-ink-400 hover:text-ink-200"
            >
              <ChevronDown size={12} className={cn("shrink-0 transition-transform", !expanded && "-rotate-90")} />
              <span className="tabular-nums">
                Reconciles: {recon.records_included ?? "—"} of {recon.total_records ?? "—"} records
                {total == null ? "" : `, ${formatGBP(total)}`}, {pct(recon.coverage_by_balance_pct)} coverage
              </span>
            </button>
          ) : (
            <div className="font-medium text-ink-300">Reconciliation &amp; coverage</div>
          )}
          {expanded && (
            <div className={cn(
              "grid grid-cols-2 gap-x-6 gap-y-0.5 tabular-nums sm:grid-cols-3",
              clean ? "mt-2 border-t border-[var(--color-line-soft)] pt-2" : "mt-1",
            )}>
              <span>Total balance: {total == null ? "—" : formatGBP(total)}</span>
              <span>Included: {included == null ? "—" : formatGBP(included)}</span>
              <span>Coverage by balance: {pct(recon.coverage_by_balance_pct)}</span>
              <span>Records: {recon.records_included ?? "—"} / {recon.total_records ?? "—"}</span>
              <span>Excluded: {formatGBP(excluded || 0)}</span>
              <span>Missing policy: {recon.missing_dimension_policy ?? "—"}</span>
            </div>
          )}
          {excluded > 0 && included != null && total != null && (
            <div className="mt-1 text-amber-300/90">
              This result covers {formatGBP(included)} of the {formatGBP(total)} funded book.{" "}
              {formatGBP(excluded)} was excluded
              {fields.length ? ` because ${fields.join(" and/or ")} was missing` : ""}.
            </div>
          )}
          {recon.filters_applied && (
            <div className="mt-1 text-ink-500">
              Filters applied: {JSON.stringify(recon.filters ?? {})}
            </div>
          )}
        </>
      )}
      {notes.length > 0 && (
        <div className="mt-1.5 border-t border-[var(--color-line-soft)] pt-1.5 text-ink-500">
          {notes.map((n) => (
            <div key={n.field}>
              <span className="text-ink-400">Source note ({n.field}):</span> {n.note}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
