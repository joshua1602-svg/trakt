import { useState } from "react";
import { AlertTriangle, ChevronDown, GitBranch } from "lucide-react";
import type { PipelineSnapshot } from "@/domain";
import { Badge } from "@/components/ui";
import { BarList, StatTile, type BarDatum, type DeltaIntent } from "@/components/pipeline/bits";
import { TimingDisclosureBanner } from "@/components/TimingDisclosureBanner";
import { cn, formatGBP } from "@/lib/utils";

/**
 * Week-on-week movement for a pipeline tile. Returns "No prior week" (neutral)
 * when the prior weekly value is genuinely absent — the UI never synthesises a
 * delta. Format controls how the magnitude is rendered (count vs GBP).
 */
function weeklyDelta(
  current: number | null | undefined,
  prior: number | null | undefined,
  format: "count" | "gbp",
): { delta: string; deltaIntent: DeltaIntent } {
  if (current == null || prior == null) return { delta: "No prior week", deltaIntent: "neutral" };
  const d = current - prior;
  if (d === 0) return { delta: "No change vs prior week", deltaIntent: "neutral" };
  const mag = format === "gbp" ? formatGBP(Math.abs(d)) : Math.abs(d).toLocaleString("en-GB");
  return { delta: `${d > 0 ? "+" : "−"}${mag} vs prior week`, deltaIntent: d > 0 ? "positive" : "negative" };
}

function dataQualityStatus(snap: PipelineSnapshot): { label: string; tone: "mint" | "amber" | "rose" } {
  const blockers = snap.dataQuality.filter((d) => d.severity === "blocker").length;
  const warnings = snap.dataQuality.filter((d) => d.severity === "warning").length;
  if (blockers) return { label: `${blockers} blocker${blockers > 1 ? "s" : ""}`, tone: "rose" };
  if (warnings) return { label: `${warnings} warning${warnings > 1 ? "s" : ""}`, tone: "amber" };
  return { label: "Clean", tone: "mint" };
}

/**
 * The governed pipeline single-source snapshot on the landing page. Separate
 * from the funded book; loads from the forecast/pipeline snapshot endpoint and
 * answers "what is in the pipeline / how much is expected to fund / when".
 */
export function PipelineSnapshotPanel({
  snapshot,
  loading,
}: {
  snapshot: PipelineSnapshot | null;
  loading?: boolean;
}) {
  const [showDiagnostics, setShowDiagnostics] = useState(false);

  if (loading && !snapshot) {
    return (
      <section className="rounded-xl border border-[var(--color-line)] bg-navy-900/40 p-5">
        <div className="h-4 w-48 animate-pulse rounded bg-navy-700/60" />
        <div className="mt-4 grid grid-cols-2 gap-3 lg:grid-cols-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="h-20 animate-pulse rounded-lg bg-navy-800/50" />
          ))}
        </div>
      </section>
    );
  }

  if (!snapshot) return null;

  if (!snapshot.ok) {
    return (
      <section className="rounded-xl border border-amber-400/20 bg-amber-400/5 p-5 text-[13px] text-amber-300/90">
        <div className="flex items-center gap-2 font-medium">
          <AlertTriangle size={15} /> Pipeline Snapshot unavailable
        </div>
        <p className="mt-1 text-amber-300/70">{snapshot.error ?? "No pipeline data for this reporting date."}</p>
      </section>
    );
  }

  const amount = snapshot.pipelineAmount ?? 0;
  const cases = snapshot.pipelineRowCount;
  const weighted = snapshot.weightedExpectedFundedAmount ?? 0;
  const avg = cases > 0 ? amount / cases : 0;

  // Week-on-week deltas from the prior weekly extract, when the backend supplies
  // one. Average ticket prior is derived only when both prior count + amount exist.
  const prior = snapshot.priorWeek ?? null;
  const priorAvg =
    prior && prior.pipelineAmount != null && prior.pipelineRowCount
      ? prior.pipelineAmount / prior.pipelineRowCount
      : null;
  const casesDelta = weeklyDelta(cases, prior?.pipelineRowCount, "count");
  const amountDelta = weeklyDelta(amount, prior?.pipelineAmount, "gbp");
  const weightedDelta = weeklyDelta(weighted, prior?.weightedExpectedFundedAmount, "gbp");
  const avgDelta = weeklyDelta(avg, priorAvg, "gbp");
  const topStage = [...snapshot.stageBreakdown].sort((a, b) => b.pipelineAmount - a.pipelineAmount)[0];
  // The "next" expected completion is the first FUTURE month (> pipeline as-of
  // month); a past month is overdue, not next. Classified backend-side.
  const summary = snapshot.expectedCompletionSummary;
  const nextMonth = summary?.nextExpectedCompletionMonth ?? null;
  const nextCompletion = nextMonth
    ? snapshot.expectedCompletionBreakdown.find((m) => m.month === nextMonth) ?? null
    : null;
  const overdueCount = summary?.overdueExpectedCompletionCount ?? 0;
  const overdueWeighted = summary?.overdueExpectedCompletionWeightedAmount ?? 0;
  const dq = dataQualityStatus(snapshot);

  const stageByAmount: BarDatum[] = snapshot.stageBreakdown.map((s) => ({
    label: s.stage,
    value: s.pipelineAmount,
    count: s.caseCount,
  }));
  const stageByCount: BarDatum[] = snapshot.stageBreakdown.map((s) => ({
    label: s.stage,
    value: s.caseCount,
  }));
  const completionByMonth: BarDatum[] = snapshot.expectedCompletionBreakdown.map((m) => ({
    label: m.month,
    value: m.weightedExpectedFundedAmount ?? 0,
    count: m.caseCount,
  }));
  const byBroker: BarDatum[] = (snapshot.brokerBreakdown ?? []).map((b) => ({
    label: b.key,
    value: b.pipelineAmount,
    count: b.caseCount,
  }));
  const byRegion: BarDatum[] = (snapshot.regionBreakdown ?? []).map((r) => ({
    label: r.key,
    value: r.pipelineAmount,
    count: r.caseCount,
  }));

  return (
    <section className="rounded-xl border border-[var(--color-line)] bg-navy-900/40 p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-navy-700/70 text-peri-300">
            <GitBranch size={17} />
          </div>
          <div className="leading-tight">
            <h2 className="text-sm font-semibold text-ink-100">Pipeline Snapshot</h2>
            <p className="text-[11px] text-ink-400">
              Origination pipeline (pre-funded) · weekly operational view · as of{" "}
              <span className="font-medium text-ink-300">
                {snapshot.pipelineAsOfDate ?? snapshot.runId}
              </span>
              {snapshot.pipelineSourceFolderDate &&
                snapshot.pipelineSourceFolderDate !== snapshot.pipelineAsOfDate && (
                  <span className="text-ink-500"> · source folder {snapshot.pipelineSourceFolderDate}</span>
                )}
            </p>
          </div>
        </div>
        <Badge tone={dq.tone}>Data quality · {dq.label}</Badge>
      </div>

      <TimingDisclosureBanner timing={snapshot.pipelineTiming} className="mt-3" />

      <div className="mt-4 grid grid-cols-2 gap-3 lg:grid-cols-3 xl:grid-cols-4">
        <StatTile label="Pipeline cases" value={cases.toLocaleString("en-GB")}
          delta={casesDelta.delta} deltaIntent={casesDelta.deltaIntent} />
        <StatTile label="Total pipeline amount" value={formatGBP(amount)}
          delta={amountDelta.delta} deltaIntent={amountDelta.deltaIntent} />
        <StatTile label="Weighted expected funded" value={formatGBP(weighted)}
          delta={weightedDelta.delta} deltaIntent={weightedDelta.deltaIntent}
          hint="probability-weighted" />
        <StatTile label="Average case amount" value={formatGBP(avg)}
          delta={avgDelta.delta} deltaIntent={avgDelta.deltaIntent} />
        {topStage && (
          <StatTile label="Top stage by amount" value={topStage.stage}
            hint={`${formatGBP(topStage.pipelineAmount)} · ${topStage.caseCount} cases`} />
        )}
        {nextCompletion ? (
          <StatTile label="Next expected completions" value={nextCompletion.month}
            hint={`${nextCompletion.caseCount} cases · ${formatGBP(nextCompletion.weightedExpectedFundedAmount ?? 0)} weighted`} />
        ) : (
          <StatTile label="Next expected completions" value="None"
            hint="no future expected completions" />
        )}
        {overdueCount > 0 && (
          <StatTile label="Overdue expected completions" value={overdueCount.toLocaleString("en-GB")}
            hint={`${formatGBP(overdueWeighted)} weighted · before as-of month`} />
        )}
      </div>

      <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Panel title="Pipeline amount by stage">
          <BarList data={stageByAmount} format="gbp" />
        </Panel>
        <Panel title="Pipeline count by stage">
          <BarList data={stageByCount} format="count" />
        </Panel>
        {completionByMonth.length > 0 && (
          <Panel title="Weighted expected funded by completion month">
            <BarList data={completionByMonth} format="gbp" />
          </Panel>
        )}
        {byBroker.length > 0 && (
          <Panel title="Pipeline amount by broker / channel">
            <BarList data={byBroker} format="gbp" />
          </Panel>
        )}
        {byRegion.length > 0 && (
          <Panel title="Pipeline amount by region">
            <BarList data={byRegion} format="gbp" />
          </Panel>
        )}
      </div>

      {snapshot.dataQuality.length > 0 && (
        <div className="mt-3">
          <button
            type="button"
            onClick={() => setShowDiagnostics((s) => !s)}
            className="inline-flex items-center gap-1.5 text-[11px] font-medium text-ink-500 hover:text-ink-300"
          >
            <ChevronDown size={13} className={cn("transition-transform", !showDiagnostics && "-rotate-90")} />
            Technical details ({snapshot.dataQuality.length})
          </button>
          {showDiagnostics && (
            <ul className="mt-1.5 list-disc space-y-0.5 rounded-lg border border-[var(--color-line-soft)] bg-navy-900/60 px-5 py-2 text-[11px] text-ink-400">
              {snapshot.dataQuality.map((d, i) => (
                <li key={i}>
                  <span className="uppercase text-ink-500">[{d.severity}]</span> {d.detail}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </section>
  );
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-[var(--color-line-soft)] bg-navy-900/50 p-3.5">
      <div className="mb-2.5 text-[11px] font-medium uppercase tracking-wider text-ink-400">{title}</div>
      {children}
    </div>
  );
}
