import type { Artifact, Reconciliation, SourceNote } from "@/domain";
import { formatGBP } from "@/lib/utils";

/**
 * The reconciliation / coverage footer shown beneath an MI artifact so a result
 * total can always be tied back to the funded-book snapshot. When some balance is
 * excluded (the operator asked to exclude missing dimensions) it says so plainly,
 * and surfaces any field-provenance source notes.
 */
function pct(v: number | null | undefined): string {
  return v == null ? "—" : `${v}%`;
}

export function ReconciliationFooter({ artifact }: { artifact: Artifact }) {
  const recon = (artifact as { reconciliation?: Reconciliation }).reconciliation;
  const notes = (artifact as { sourceNotes?: SourceNote[] }).sourceNotes ?? [];
  if (!recon && notes.length === 0) return null;

  const excluded = recon?.balance_excluded_missing ?? 0;
  const included = recon?.balance_included ?? null;
  const total = recon?.total_balance ?? null;
  const fields = recon?.missing_dimension_fields ?? [];

  return (
    <div className="mt-3 rounded-lg border border-[var(--color-line-soft)] bg-navy-900/50 px-3 py-2 text-[11px] text-ink-400">
      {recon && (
        <>
          <div className="font-medium text-ink-300">Reconciliation &amp; coverage</div>
          <div className="mt-1 grid grid-cols-2 gap-x-6 gap-y-0.5 tabular-nums sm:grid-cols-3">
            <span>Total balance: {total == null ? "—" : formatGBP(total)}</span>
            <span>Included: {included == null ? "—" : formatGBP(included)}</span>
            <span>Coverage by balance: {pct(recon.coverage_by_balance_pct)}</span>
            <span>Records: {recon.records_included ?? "—"} / {recon.total_records ?? "—"}</span>
            <span>Excluded: {formatGBP(excluded || 0)}</span>
            <span>Missing policy: {recon.missing_dimension_policy ?? "—"}</span>
          </div>
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
