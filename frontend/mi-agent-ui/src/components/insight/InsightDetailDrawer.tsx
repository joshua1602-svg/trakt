/**
 * Phase 2A — the light drill panel.
 *
 * Contained: a side drawer over the page. No route change, no page reload, no
 * chart remount, and — the point of the phase — NO second calculation. It
 * renders the SAME `MovementDetail` object the tooltip already has, so the
 * hover and the panel can never disagree.
 *
 * It shows more of that object, not different information: the full ranked
 * contributor lists with share and case count, the component breakdown, and the
 * methodology and dates the tooltip only summarises. It deliberately does not
 * list cases — this release builds no loan-level explorer, and the payload
 * carries no case-level rows to build one from.
 */

import { useEffect, useRef } from "react";
import { X } from "lucide-react";
import type { MovementContributor, MovementDetail } from "@/domain";
import { COMPONENT_LABEL, MOVEMENT_COMPONENTS } from "@/domain";
import { cn, formatGBP } from "@/lib/utils";

function signed(v: number): string {
  return `${v >= 0 ? "+" : "−"}${formatGBP(Math.abs(v), { compact: true })}`;
}

function Contributors({ title, rows, total }: {
  title: string; rows: MovementContributor[] | undefined; total: number | null;
}) {
  const items = rows ?? [];
  return (
    <section className="mt-4">
      <h4 className="text-[11px] font-semibold uppercase tracking-wide text-ink-400">
        {title}
      </h4>
      {items.length === 0 ? (
        <p className="mt-1 text-[11px] text-ink-500">No contributors for this week.</p>
      ) : (
        <table className="mt-1 w-full text-[11px]">
          <thead>
            <tr className="text-ink-500">
              <th className="py-1 text-left font-normal">Name</th>
              <th className="py-1 text-right font-normal">Contribution</th>
              <th className="py-1 text-right font-normal">Share</th>
              <th className="py-1 text-right font-normal">Cases</th>
            </tr>
          </thead>
          <tbody>
            {items.map((c) => (
              <tr key={c.name} className="border-t border-[var(--color-line-soft)]">
                <td className="max-w-[180px] truncate py-1 text-ink-300" title={c.name}>
                  {c.name}
                </td>
                <td className={cn("py-1 text-right tabular-nums",
                  c.amount < 0 ? "text-[#eb6f6f]" : "text-[#5ec6b8]")}>
                  {signed(c.amount)}
                </td>
                <td className="py-1 text-right tabular-nums text-ink-400">
                  {c.share_of_change_pct == null ? "—" : `${c.share_of_change_pct.toFixed(1)}%`}
                </td>
                <td className="py-1 text-right tabular-nums text-ink-400">{c.case_count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {total != null && items.length > 0 && (
        <p className="mt-1 text-[10px] text-ink-500">
          Shares are of the total movement ({signed(total)}); contributions are
          changes, not current balances.
        </p>
      )}
    </section>
  );
}

export function InsightDetailDrawer({ detail, loading, onClose }: {
  detail: MovementDetail | null;
  loading?: boolean;
  onClose: () => void;
}) {
  const panel = useRef<HTMLDivElement>(null);

  // Escape closes, and focus moves into the panel so a keyboard user is not
  // stranded behind the chart they just opened it from.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", onKey);
    panel.current?.focus();
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  const head = detail?.available ? detail.headline_metric : null;

  return (
    <div className="fixed inset-0 z-40 flex justify-end" data-testid="insight-drawer">
      <div className="flex-1 bg-black/40" onClick={onClose} aria-hidden />
      <div
        ref={panel}
        role="dialog"
        aria-modal="true"
        aria-label="Movement details"
        tabIndex={-1}
        className="w-[420px] max-w-full overflow-y-auto border-l border-[var(--color-line)] bg-[#0f1626] p-4 shadow-xl outline-none"
      >
        <div className="flex items-start justify-between gap-2">
          <div>
            <h3 className="text-[13px] font-semibold text-ink-100">
              {head?.label ?? "Movement details"}
            </h3>
            {detail?.as_of_date && (
              <p className="text-[10px] text-ink-500">
                Pipeline {detail.as_of_date}
                {detail.comparison_date && ` vs ${detail.comparison_date}`}
                {detail.scope && ` · scope ${detail.scope}`}
              </p>
            )}
          </div>
          <button type="button" onClick={onClose} aria-label="Close details"
            className="rounded-md p-1 text-ink-400 hover:bg-navy-800 hover:text-ink-100">
            <X size={14} />
          </button>
        </div>

        {loading && (
          <p className="mt-4 text-[11px] text-ink-500" data-testid="insight-drawer-loading">
            Loading movement detail…
          </p>
        )}

        {!loading && !head && (
          <p className="mt-4 text-[11px] text-ink-500" data-testid="insight-drawer-unavailable">
            {detail?.reason ?? "No movement detail for this point."}
          </p>
        )}

        {!loading && head && (
          <>
            <div className="mt-3 rounded-md border border-[var(--color-line-soft)] bg-navy-900/50 p-3">
              <div className="text-[18px] font-semibold tabular-nums text-ink-100">
                {formatGBP(head.value, { compact: true })}
              </div>
              <div className={cn("text-[12px] tabular-nums",
                head.change < 0 ? "text-[#eb6f6f]" : "text-[#5ec6b8]")}>
                {signed(head.change)}
                {head.change_pct != null && ` / ${head.change_pct >= 0 ? "+" : ""}${head.change_pct.toFixed(1)}%`}
                <span className="text-ink-500"> vs prior week</span>
              </div>
              {detail?.counts && (
                <div className="mt-1 text-[11px] tabular-nums text-ink-400">
                  {detail.counts.current.toLocaleString("en-GB")} cases
                  <span className="text-ink-500">
                    {" "}({detail.counts.change >= 0 ? "+" : "−"}
                    {Math.abs(detail.counts.change).toLocaleString("en-GB")})
                  </span>
                </div>
              )}
            </div>

            <Contributors title="Broker contributors"
              rows={detail?.contributors?.brokers} total={head.change} />
            <Contributors title="Regional contributors"
              rows={detail?.contributors?.regions} total={head.change} />

            {detail?.components && (
              <section className="mt-4" data-testid="insight-drawer-components">
                <h4 className="text-[11px] font-semibold uppercase tracking-wide text-ink-400">
                  What moved
                </h4>
                <ul className="mt-1 space-y-0.5">
                  {MOVEMENT_COMPONENTS.map((k) => {
                    const c = detail.components?.[k];
                    if (!c) return null;
                    return (
                      <li key={k} className="flex items-baseline justify-between gap-3 text-[11px]">
                        <span className="text-ink-500">{COMPONENT_LABEL[k]}</span>
                        <span className="tabular-nums text-ink-300">
                          {signed(c.amount)}
                          <span className="text-ink-500"> · {c.cases} cases</span>
                        </span>
                      </li>
                    );
                  })}
                </ul>
              </section>
            )}

            {detail?.methodology && (
              <section className="mt-4 border-t border-[var(--color-line-soft)] pt-3
                text-[10px] leading-snug text-ink-500"
                data-testid="insight-drawer-methodology">
                <p>{detail.methodology.metric_definition}</p>
                <p className="mt-1">
                  Basis: {detail.methodology.movement_basis} · attribution:{" "}
                  {detail.methodology.attribution} · v{detail.methodology.version}
                </p>
                {Object.entries(detail.methodology.dimension_reassignments ?? {})
                  .filter(([, n]) => n > 0).length > 0 && (
                  <p className="mt-1 text-amber-300/80">
                    {Object.entries(detail.methodology.dimension_reassignments ?? {})
                      .filter(([, n]) => n > 0)
                      .map(([d, n]) => `${n} case(s) changed ${d}`)
                      .join("; ")} between these two weeks — attribution follows
                    the current period, so those contributions sit with the new value.
                  </p>
                )}
                {detail.sources?.current && (
                  <p className="mt-1">
                    Source: {detail.sources.current}
                    {detail.sources.comparison && ` vs ${detail.sources.comparison}`}
                  </p>
                )}
              </section>
            )}
          </>
        )}
      </div>
    </div>
  );
}
