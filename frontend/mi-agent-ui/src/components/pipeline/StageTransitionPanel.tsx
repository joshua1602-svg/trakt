/**
 * Sprint 2 — Pipeline Stage Movement.
 *
 * Answers the one question a stage-stock chart cannot: *what happened to
 * pipeline cases between the prior snapshot and the latest one?* An OFFER stock
 * falling from three cases to one is a net -2 whether three left and one
 * arrived or two left and none did. This panel shows which.
 *
 * IT COMPUTES NOTHING. Every count, amount, outcome and residual is read from
 * the governed `PIPELINE_STAGE_TRANSITION` payload, fetched through the SAME
 * `/mi/insight/movement-detail` route the movement hover already uses. The
 * panel orders, labels and formats — nothing else. The deck renders the same
 * payload, so the two channels cannot disagree.
 *
 * Four populations are kept visually apart on purpose, because merging any two
 * of them would state a different metric:
 *
 *   TRUE TRANSITIONS  in both snapshots, at different stages
 *   NEW ARRIVALS      no prior stage at all — never drawn as if it had one
 *   STAYERS           same stage in both, carrying any amount amendment
 *   DEPARTURES        gone from the latest extract, named only by evidence
 */

import { useMemo, useState } from "react";
import type { AgentClient } from "@/api";
import type {
  StageArrivalRow, StageDepartureRow, StageStayerRow, StageTransitionDetail,
  StageTransitionRow,
} from "@/domain";
import { UNCLASSIFIED_DEPARTURE } from "@/domain";
import { useStageTransitionDetail } from "@/hooks/useStageTransitionDetail";
import { cn, formatGBP } from "@/lib/utils";

/** Display spellings for the governed canonical stage tokens. Presentation
 * only: the tokens themselves are the engine's and are never re-derived. */
const STAGE_LABEL: Record<string, string> = {
  KFI: "KFI", APPLICATION: "Application", OFFER: "Offer",
  COMPLETED: "Completed", WITHDRAWN: "Withdrawn", UNKNOWN: "Unknown",
};

function stageLabel(stage: string | null | undefined): string {
  const key = String(stage ?? "").toUpperCase();
  return STAGE_LABEL[key] ?? (key ? key[0] + key.slice(1).toLowerCase() : "—");
}

function signedGBP(value: number): string {
  if (!value) return "No change";
  return `${value > 0 ? "+" : "−"}${formatGBP(Math.abs(value))}`;
}

function cases(n: number): string {
  return `${n.toLocaleString()} ${n === 1 ? "case" : "cases"}`;
}

/**
 * A departure named by the evidence, never by assumption.
 *
 * Where the prior extract recorded a governed terminal stage, that stage IS the
 * outcome (so the engine's source and outcome are the same value and naming it
 * once is enough). Where it recorded nothing, the reader is told exactly that
 * — absence from the latest extract is not evidence of a withdrawal.
 */
export function departureLabel(row: StageDepartureRow): string {
  const outcome = String(row.governed_outcome ?? "");
  if (!outcome || outcome === UNCLASSIFIED_DEPARTURE) {
    return `Left from ${stageLabel(row.source_stage)} — unclassified`;
  }
  return `Left after ${stageLabel(outcome)}`;
}

type Measure = "count" | "amount";

function Row({ label, value, muted, tone }: {
  label: string; value: string; muted?: boolean;
  tone?: "up" | "down" | "flat";
}) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-1">
      <span className={cn("text-[12px]", muted ? "text-ink-500" : "text-ink-200")}>
        {label}
      </span>
      <span className={cn(
        "shrink-0 font-mono text-[12px] tabular-nums",
        tone === "up" ? "text-mint-400"
          : tone === "down" ? "text-rose-400" : "text-ink-100",
      )}>
        {value}
      </span>
    </div>
  );
}

function Block({ title, hint, children, empty, testId }: {
  title: string; hint?: string; children: React.ReactNode; empty?: string;
  testId: string;
}) {
  return (
    <div data-testid={testId}
      className="rounded-lg border border-navy-600/70 bg-navy-800/60 p-3">
      <div className="text-[11px] font-semibold uppercase tracking-wider text-ink-400">
        {title}
      </div>
      {hint && <div className="mt-0.5 text-[10px] text-ink-500">{hint}</div>}
      <div className="mt-2 divide-y divide-navy-700/60">
        {children}
      </div>
      {empty && <div className="mt-1 text-[11px] text-ink-500">{empty}</div>}
    </div>
  );
}

export interface StageTransitionPanelProps {
  client: AgentClient;
  portfolioId: string;
  portfolioContext?: string;
  /** Off unless the deployment has enabled the enhanced insight layer. */
  enabled: boolean;
  /** The week to explain; omit for the latest governed pair. */
  asOf?: string | null;
}

export function StageTransitionPanel(props: StageTransitionPanelProps) {
  const { detail, loading, unavailable } = useStageTransitionDetail(props);
  const [measure, setMeasure] = useState<Measure>("count");

  if (!props.enabled) return null;
  if (loading && !detail) {
    return (
      // A distinct id: "still loading" and "loaded" are different states, and a
      // consumer that could not tell them apart would read an empty panel as an
      // answer.
      <div data-testid="stage-transitions-loading" className={SHELL}>
        <div className="text-[12px] text-ink-500">Loading stage movement…</div>
      </div>
    );
  }
  // The engine's own typed availability, surfaced in its own words. The panel
  // never decides this for itself, and never falls back to an empty matrix —
  // "no rows" and "we declined to answer" are different statements.
  if (unavailable || !detail || !detail.available) {
    return (
      <div data-testid="stage-transitions-unavailable" className={SHELL}>
        <Heading detail={detail} />
        <div className="mt-1 text-[12px] text-ink-500">
          {detail?.reason ?? "Stage movement is not available for this window."}
        </div>
      </div>
    );
  }
  return <Available detail={detail} measure={measure} onMeasure={setMeasure} />;
}

const SHELL = "rounded-xl border border-[var(--color-line)] bg-navy-900/40 p-4 lg:col-span-2";

function Heading({ detail }: { detail: StageTransitionDetail | null }) {
  return (
    <div className="text-[12px] font-semibold text-ink-200">
      Pipeline stage movement
      {detail?.comparison_date && detail?.as_of_date && (
        <span className="ml-2 font-normal text-ink-500">
          {detail.comparison_date} → {detail.as_of_date}
        </span>
      )}
    </div>
  );
}

function Available({ detail, measure, onMeasure }: {
  detail: StageTransitionDetail;
  measure: Measure;
  onMeasure: (m: Measure) => void;
}) {
  // Ordering only — the values are the engine's, in the order it published.
  const moves: StageTransitionRow[] = detail.transitions ?? [];
  const arrivals: StageArrivalRow[] = detail.new_arrivals ?? [];
  const stayers: StageStayerRow[] = detail.stayers ?? [];
  const departures: StageDepartureRow[] = detail.departures ?? [];
  const recon = detail.reconciliation;

  const value = useMemo(
    () => (n: number, gbp: number) => (measure === "count" ? cases(n) : formatGBP(gbp)),
    [measure],
  );

  return (
    <div data-testid="stage-transitions" className={SHELL}>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <Heading detail={detail} />
        <div role="tablist" aria-label="Stage movement measure"
          className="inline-flex items-center gap-0.5 rounded-lg border border-[var(--color-line)] bg-navy-900/60 p-0.5">
          {([["count", "Cases"], ["amount", "Value"]] as const).map(([m, label]) => (
            <button key={m} type="button" role="tab" aria-selected={measure === m}
              onClick={() => onMeasure(m)}
              className={cn(
                "rounded-md px-2 py-0.5 text-[11px]",
                measure === m ? "bg-navy-700 text-ink-100" : "text-ink-400",
              )}>
              {label}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
        <Block title="Moved stage" testId="stx-moves"
          hint="In both snapshots, at a different stage"
          empty={moves.length ? undefined : "No case changed stage."}>
          {moves.map((m) => (
            <Row key={`${m.source_stage}->${m.destination_stage}`}
              label={`${stageLabel(m.source_stage)} → ${stageLabel(m.destination_stage)}`}
              value={value(m.case_count, m.latest_amount)} />
          ))}
        </Block>

        {/* A new arrival has NO prior stage. It is never rendered as
            "KFI → KFI" or from any other synthetic source. */}
        <Block title="New arrivals" testId="stx-arrivals"
          hint="Not in the prior snapshot at all"
          empty={arrivals.length ? undefined : "No new cases arrived."}>
          {arrivals.map((a) => (
            <Row key={a.destination_stage}
              label={`New into ${stageLabel(a.destination_stage)}`}
              value={value(a.case_count, a.latest_amount)} />
          ))}
        </Block>

        {/* An amendment leaves the case exactly where it was. It must never be
            drawn as an exit plus an arrival. */}
        <Block title="Stayed in place" testId="stx-stayers"
          hint="Same stage in both; value may have been amended"
          empty={stayers.length ? undefined : "No case stayed at the same stage."}>
          {stayers.map((s) => (
            <Row key={s.stage} label={stageLabel(s.stage)}
              value={measure === "count" ? cases(s.case_count)
                : signedGBP(s.amount_change)}
              tone={measure === "amount"
                ? (s.amount_change > 0 ? "up" : s.amount_change < 0 ? "down" : "flat")
                : undefined} />
          ))}
        </Block>

        <Block title="Left the pipeline" testId="stx-departures"
          hint="Not in the latest snapshot"
          empty={departures.length ? undefined : "No case left the pipeline."}>
          {departures.map((dep) => (
            <Row key={`${dep.source_stage}|${dep.governed_outcome}`}
              label={departureLabel(dep)}
              muted={dep.governed_outcome === UNCLASSIFIED_DEPARTURE}
              value={value(dep.case_count, dep.prior_amount)} />
          ))}
        </Block>
      </div>

      {recon && recon.by_stage?.length > 0 && (
        <div className="mt-3 overflow-x-auto">
          <table className="w-full min-w-[560px] text-[11px]"
            data-testid="stage-transitions-reconciliation">
            <caption className="sr-only">
              Opening to closing case reconciliation by pipeline stage
            </caption>
            <thead className="text-ink-500">
              <tr className="text-left">
                <th scope="col" className="py-1 pr-2 font-medium">Stage</th>
                {["Opening", "New", "In", "Out", "Left", "Closing"].map((h) => (
                  <th key={h} scope="col" className="py-1 pl-2 text-right font-medium">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="font-mono tabular-nums text-ink-200">
              {recon.by_stage.map((r) => (
                <tr key={r.stage} className="border-t border-navy-700/60">
                  <th scope="row" className="py-1 pr-2 text-left font-sans font-normal text-ink-300">
                    {stageLabel(r.stage)}
                  </th>
                  <td className="py-1 pl-2 text-right">{r.opening_case_count}</td>
                  <td className="py-1 pl-2 text-right">{r.new_arrivals}</td>
                  <td className="py-1 pl-2 text-right">{r.transitions_in}</td>
                  <td className="py-1 pl-2 text-right">{r.transitions_out}</td>
                  <td className="py-1 pl-2 text-right">{r.departures}</td>
                  <td className="py-1 pl-2 text-right text-ink-100">{r.closing_case_count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* The residuals are disclosed, never hidden: a non-zero one is a finding
          about the data, and a diagram that does not add up must say so. */}
      <div className="mt-2 text-[10px] leading-relaxed text-ink-500">
        Cases matched on {detail.identifier ?? "the governed case identifier"}
        {detail.counts && ` (${detail.counts.comparison} prior, ${detail.counts.current} latest)`}.
        Every case is classified once. Reconciliation residual{" "}
        {recon?.count_reconciliation_residual ?? "—"} cases /{" "}
        {recon?.amount_reconciliation_residual ?? "—"} by value. A change in
        amount does not change case identity.
      </div>
    </div>
  );
}
