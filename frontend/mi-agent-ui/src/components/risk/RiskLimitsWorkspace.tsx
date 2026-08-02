/**
 * Funded → Risk Limits — three-state concentration workspace.
 *
 * One governed evaluation (`/mi/concentration-tests`) supplies everything:
 * the contractual **Funded** position, the **Expected Forecast** (existing
 * completion-trend model over the weekly-snapshot observation window) and the
 * **Full Pipeline** maximum-exposure stress, plus service-ranked emerging
 * risks and forecast methodology provenance.
 *
 * The UI renders governed results verbatim: it never recalculates a test,
 * never re-ranks risks, never treats an unavailable value as zero, and never
 * presents the stress state as a prediction. Status is always conveyed by
 * label and glyph as well as colour. Layout follows
 * docs/design/risk_limits/ (wireframes 01–07).
 */

import { useEffect, useMemo, useState } from "react";
import { Search } from "lucide-react";
import type { AgentClient } from "@/api/AgentClient";
import type {
  ConcentrationTestStatus,
  ConcentrationTestsSnapshot,
} from "@/domain";
import { Badge, Card } from "@/components/ui";
import { ConcentrationDetailPanel } from "./ConcentrationDetailPanel";
import { MethodologyBlock } from "./MethodologyBlock";
import {
  RISK_CHIP,
  STATUS_GLYPH,
  STATUS_LABEL,
  STATUS_TONE,
  categoryLabel,
  formatChange,
  formatDate,
  formatValue,
  operatorGlyph,
  riskCategoryByTest,
  riskSortKey,
  statePhrase,
} from "./concentrationShared";

const STATUS_ORDER: Record<ConcentrationTestStatus, number> = {
  breach: 0,
  warning: 1,
  unavailable: 2,
  insufficient_data: 3,
  pending_effective_date: 4,
  expired: 5,
  pass: 6,
};

type SortKey = "risk" | "status" | "name" | "headroom" | "change";

function SummaryTile({
  label,
  sublabel,
  value,
  tone,
  testId,
}: {
  label: string;
  sublabel?: string;
  value: string | number;
  tone?: "mint" | "amber" | "rose" | "neutral";
  testId?: string;
}) {
  const toneClass =
    tone === "rose"
      ? "text-rose-400"
      : tone === "amber"
        ? "text-amber-400"
        : tone === "mint"
          ? "text-mint-400"
          : "text-ink-100";
  return (
    <div
      data-testid={testId}
      className="rounded-lg border border-[var(--color-line-soft)] bg-navy-900/50 px-3 py-2"
    >
      <p className="text-[10px] uppercase tracking-wider text-ink-500">{label}</p>
      <p className={`font-mono text-[18px] tabular-nums ${toneClass}`}>{value}</p>
      {sublabel && <p className="text-[9px] text-ink-500">{sublabel}</p>}
    </div>
  );
}

function SourceBanner({ snapshot }: { snapshot: ConcentrationTestsSnapshot }) {
  if (snapshot.source === "approved_configuration") {
    return (
      <p
        role="note"
        data-testid="concentration-source-banner"
        className="rounded-lg border border-[var(--color-line-soft)] bg-navy-900/50 px-3 py-2 text-[11px] text-ink-400"
      >
        Approved configuration v{snapshot.configurationVersion}
        {snapshot.activatedBy && <> · activated by {snapshot.activatedBy}</>}
        {snapshot.activatedAt && <> on {formatDate(snapshot.activatedAt)}</>} · evaluated
        against the governed funded snapshot
        {snapshot.reportingDate && <> at {formatDate(snapshot.reportingDate)}</>}.
        {snapshot.openProposals ? (
          <span className="ml-1 text-amber-300/90">
            {snapshot.openProposals} proposal(s) from onboarding still await review or
            approval.
          </span>
        ) : null}
      </p>
    );
  }
  if (snapshot.source === "legacy_extracted") {
    return (
      <p
        role="note"
        data-testid="concentration-source-banner"
        className="rounded-lg border border-amber-400/20 bg-amber-400/5 px-3 py-2 text-[11px] text-amber-300/90"
      >
        Showing limits extracted from documents for continuity — they are NOT
        operator-approved concentration tests. Review and approve proposals through the
        OCC concentration-test workflow to govern this view.
      </p>
    );
  }
  return null;
}

function ForecastBanner({ snapshot }: { snapshot: ConcentrationTestsSnapshot }) {
  const [open, setOpen] = useState(false);
  const forecast = snapshot.forecast;
  const statesAvailable = Boolean(snapshot.states?.available);
  if (snapshot.source !== "approved_configuration") return null;
  if (!statesAvailable) {
    const reason =
      snapshot.states?.reason ?? forecast?.reason ?? "no governed pipeline forecast.";
    return (
      <p
        role="note"
        data-testid="forecast-unavailable-banner"
        className="rounded-lg border border-[var(--color-line-soft)] bg-navy-900/50 px-3 py-2 text-[11px] text-ink-400"
      >
        Expected Forecast is unavailable: {reason} Funded results are unaffected.
      </p>
    );
  }
  return (
    <div
      data-testid="forecast-provenance-banner"
      className="rounded-lg border border-[var(--color-line-soft)] bg-navy-900/50 px-3 py-2 text-[11px] text-ink-400"
    >
      <p role="note">
        Expected Forecast: completion-trend model · window{" "}
        {formatDate(forecast?.observationWindowStart)} →{" "}
        {formatDate(forecast?.observationWindowEnd)} ({forecast?.weeklyExtractsUsed}{" "}
        weekly extracts, {forecast?.trackedCaseCount} cases,{" "}
        {forecast?.observedCompletionCount} completions) ·{" "}
        <button
          type="button"
          className="text-peri-200 underline-offset-2 hover:underline"
          aria-expanded={open}
          onClick={() => setOpen((v) => !v)}
        >
          {open ? "Hide methodology" : "View methodology"}
        </button>
      </p>
      {(forecast?.stagesUsingConfigFallback?.length ?? 0) > 0 && (
        <p className="mt-1 text-amber-300/90">
          Stage(s) {forecast?.stagesUsingConfigFallback?.join(", ")} fall back to
          configured assumptions — the observed sample is below the sufficiency floor.
        </p>
      )}
      {open && forecast && (
        <div className="mt-2">
          <MethodologyBlock forecast={forecast} />
        </div>
      )}
    </div>
  );
}

function EmergingRisks({
  snapshot,
  onOpen,
}: {
  snapshot: ConcentrationTestsSnapshot;
  onOpen: (testId: string) => void;
}) {
  const risks = snapshot.emergingRisks ?? [];
  if (snapshot.source !== "approved_configuration" || risks.length === 0) return null;
  return (
    <Card className="p-3" testId="emerging-risks">
      <h3 className="text-[12px] font-semibold text-ink-100">Emerging risks</h3>
      <ol className="mt-1 space-y-1.5">
        {risks.map((r, i) => {
          const chip = RISK_CHIP[r.category] ?? RISK_CHIP.ok;
          return (
            <li key={`${r.category}-${r.testId ?? i}`} className="flex items-start gap-2 text-[12px]">
              <span className="font-mono text-[10px] text-ink-500">{i + 1}</span>
              <Badge tone={chip.tone} className="shrink-0">
                {chip.label}
              </Badge>
              <span className="text-ink-300">
                {r.statement}
                {r.testId && (
                  <button
                    type="button"
                    className="ml-2 text-[11px] text-peri-200 underline-offset-2 hover:underline"
                    onClick={() => onOpen(r.testId!)}
                  >
                    Open test
                  </button>
                )}
              </span>
            </li>
          );
        })}
      </ol>
    </Card>
  );
}

export function RiskLimitsWorkspace({
  client,
  portfolioId,
  portfolioContext,
}: {
  client: AgentClient;
  portfolioId: string;
  portfolioContext?: string;
}) {
  const [snapshot, setSnapshot] = useState<ConcentrationTestsSnapshot | null>(null);
  const [loading, setLoading] = useState(false);
  const [failed, setFailed] = useState(false);
  const [search, setSearch] = useState("");
  const [category, setCategory] = useState("all");
  const [status, setStatus] = useState("all");
  const [sortKey, setSortKey] = useState<SortKey | null>(null);
  const [expectedOnly, setExpectedOnly] = useState(false);
  const [stressOnly, setStressOnly] = useState(false);
  const [showPrior, setShowPrior] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setFailed(false);
    client
      .getConcentrationTests(portfolioId, portfolioContext)
      .then((d) => {
        if (!cancelled) setSnapshot(d);
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
  }, [client, portfolioId, portfolioContext]);

  const tests = useMemo(() => snapshot?.tests ?? [], [snapshot]);
  const statesAvailable = Boolean(snapshot?.states?.available);
  const categoryByTest = useMemo(
    () => riskCategoryByTest(snapshot?.emergingRisks),
    [snapshot],
  );
  const categories = useMemo(
    () => Array.from(new Set(tests.map((t) => t.category))).sort(),
    [tests],
  );
  const effectiveSort: SortKey = sortKey ?? (statesAvailable ? "risk" : "status");

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    const rows = tests.filter((t) => {
      if (category !== "all" && t.category !== category) return false;
      if (status !== "all") {
        if (status === "unavailable") {
          if (!["unavailable", "insufficient_data"].includes(t.status)) return false;
        } else if (t.status !== status) return false;
      }
      if (expectedOnly && !t.expectedBreach) return false;
      if (stressOnly && !t.fullPipelineBreach) return false;
      if (q && !t.displayName.toLowerCase().includes(q)) return false;
      return true;
    });
    const sorted = [...rows];
    sorted.sort((a, b) => {
      if (effectiveSort === "name") return a.displayName.localeCompare(b.displayName);
      if (effectiveSort === "headroom") {
        const ah = (statesAvailable ? a.expected?.headroom : null) ?? a.headroom ?? Number.POSITIVE_INFINITY;
        const bh = (statesAvailable ? b.expected?.headroom : null) ?? b.headroom ?? Number.POSITIVE_INFINITY;
        return ah - bh;
      }
      if (effectiveSort === "change") {
        const ac = a.changeFundedToExpected ?? a.absoluteChange ?? 0;
        const bc = b.changeFundedToExpected ?? b.absoluteChange ?? 0;
        return Math.abs(bc) - Math.abs(ac);
      }
      if (effectiveSort === "risk") {
        const [ar, ah, an] = riskSortKey(a, categoryByTest);
        const [br, bh, bn] = riskSortKey(b, categoryByTest);
        return ar - br || ah - bh || an.localeCompare(bn);
      }
      return (
        STATUS_ORDER[a.status] - STATUS_ORDER[b.status] ||
        (a.headroom ?? Number.POSITIVE_INFINITY) - (b.headroom ?? Number.POSITIVE_INFINITY)
      );
    });
    return sorted;
  }, [tests, search, category, status, expectedOnly, stressOnly, effectiveSort,
      categoryByTest, statesAvailable]);

  const selected = tests.find((t) => t.testId === selectedId) ?? null;

  if (loading && !snapshot) {
    return (
      <Card className="p-4" testId="risk-limits-panel">
        <p className="text-[12px] text-ink-500">Loading concentration tests…</p>
      </Card>
    );
  }
  if (failed && !snapshot) {
    return (
      <Card className="p-4" testId="risk-limits-panel">
        <p
          className="rounded-lg border border-amber-400/20 bg-amber-400/5 px-3 py-2 text-[12px] text-amber-300/90"
          data-testid="concentration-error"
        >
          The Risk Limits service could not be reached. The last governed results will
          reappear when it recovers — nothing is estimated in the meantime.
        </p>
      </Card>
    );
  }
  if (!snapshot) return null;

  if (!snapshot.available || tests.length === 0) {
    return (
      <Card className="p-4 space-y-2" testId="risk-limits-panel">
        <h2 className="text-[14px] font-semibold text-ink-100">Risk Limits</h2>
        <p
          className="rounded-lg border border-[var(--color-line-soft)] bg-navy-900/50 px-3 py-2 text-[12px] text-ink-400"
          data-testid="concentration-empty"
        >
          No active concentration tests yet. The onboarding concentration-test request
          feeds proposals into the OCC review workflow; tests appear here once an
          operator approves and activates them.
          {snapshot.openProposals ? (
            <span className="ml-1 text-amber-300/90">
              {snapshot.openProposals} proposal(s) currently await review.
            </span>
          ) : null}
          {snapshot.lineage?.note ? (
            <span className="block text-ink-500">{String(snapshot.lineage.note)}</span>
          ) : null}
        </p>
      </Card>
    );
  }

  const s = snapshot.summary;

  return (
    <div className="space-y-3" data-testid="risk-limits-panel">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-[14px] font-semibold text-ink-100">Risk Limits</h2>
        <p className="text-[11px] text-ink-500">
          Reporting date {formatDate(snapshot.reportingDate)}
          {s.priorAvailable && <> · prior {formatDate(s.priorReportingDate)}</>}
          {snapshot.evaluatedAt && <> · evaluated {formatDate(snapshot.evaluatedAt)}</>}
          {!snapshot.fundedDataAvailable && (
            <span className="ml-1 text-amber-300/90">· funded data unavailable</span>
          )}
        </p>
      </div>

      <SourceBanner snapshot={snapshot} />
      <ForecastBanner snapshot={snapshot} />

      {/* Portfolio summary — grouped by state, left → right. */}
      <div
        className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6"
        data-testid="concentration-summary"
      >
        <SummaryTile
          label="Funded breaches"
          sublabel="contractual"
          value={s.breaches}
          tone={s.breaches ? "rose" : "mint"}
          testId="concentration-breaches"
        />
        <SummaryTile
          label="Expected breaches"
          sublabel="prediction"
          value={statesAvailable ? (s.expectedBreaches ?? 0) : "—"}
          tone={s.expectedBreaches ? "rose" : "neutral"}
          testId="concentration-expected-breaches"
        />
        <SummaryTile
          label="Full pipeline"
          sublabel="stress — max exposure"
          value={statesAvailable ? (s.fullPipelineBreaches ?? 0) : "—"}
          tone="neutral"
          testId="concentration-stress-breaches"
        />
        <SummaryTile
          label="Expected warnings"
          value={statesAvailable ? (s.expectedWarnings ?? 0) : "—"}
          tone={s.expectedWarnings ? "amber" : "neutral"}
        />
        <SummaryTile
          label="Deteriorating"
          sublabel="vs prior period"
          value={s.priorAvailable ? s.deteriorations : "—"}
          tone={s.deteriorations ? "amber" : "neutral"}
          testId="concentration-deteriorations"
        />
        <SummaryTile
          label="Unavailable"
          sublabel="never shown pass"
          value={s.unavailable}
          tone="neutral"
        />
      </div>
      {s.unavailable > 0 && (
        <p
          className="rounded-lg border border-[var(--color-line-soft)] bg-navy-900/50 px-3 py-1.5 text-[11px] text-ink-400"
          data-testid="concentration-unavailable-note"
        >
          {s.unavailable} test(s) could not be evaluated and are shown as Unavailable —
          never as passing.
        </p>
      )}

      <EmergingRisks snapshot={snapshot} onOpen={setSelectedId} />

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-2 text-[12px]">
        <label className="relative">
          <Search
            size={12}
            aria-hidden
            className="pointer-events-none absolute left-2 top-1/2 -translate-y-1/2 text-ink-500"
          />
          <input
            type="search"
            aria-label="Search tests"
            placeholder="Search tests"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-44 rounded-md border border-[var(--color-line)] bg-navy-900/70 py-1 pl-6 pr-2 text-[12px] text-ink-300 placeholder:text-ink-500"
          />
        </label>
        <select
          aria-label="Filter by category"
          value={category}
          onChange={(e) => setCategory(e.target.value)}
          className="rounded-md border border-[var(--color-line)] bg-navy-900/70 px-2 py-1 text-[12px] text-ink-300"
        >
          <option value="all">All categories</option>
          {categories.map((c) => (
            <option key={c} value={c}>
              {categoryLabel(c)}
            </option>
          ))}
        </select>
        <select
          aria-label="Filter by status"
          value={status}
          onChange={(e) => setStatus(e.target.value)}
          className="rounded-md border border-[var(--color-line)] bg-navy-900/70 px-2 py-1 text-[12px] text-ink-300"
        >
          <option value="all">All statuses</option>
          <option value="breach">Breach</option>
          <option value="warning">Warning</option>
          <option value="pass">Pass</option>
          <option value="unavailable">Unavailable</option>
        </select>
        <select
          aria-label="Sort tests"
          value={effectiveSort}
          onChange={(e) => setSortKey(e.target.value as SortKey)}
          className="rounded-md border border-[var(--color-line)] bg-navy-900/70 px-2 py-1 text-[12px] text-ink-300"
        >
          {statesAvailable && <option value="risk">Sort: expected risk</option>}
          <option value="status">Sort: severity</option>
          <option value="headroom">
            {statesAvailable ? "Sort: expected headroom" : "Sort: headroom"}
          </option>
          <option value="change">Sort: movement</option>
          <option value="name">Sort: name</option>
        </select>
        {statesAvailable && (
          <>
            <button
              type="button"
              aria-pressed={expectedOnly}
              onClick={() => setExpectedOnly((v) => !v)}
              className={`rounded-md border px-2 py-1 text-[11px] ${
                expectedOnly
                  ? "border-peri-400/40 bg-navy-800 text-peri-200"
                  : "border-[var(--color-line)] text-ink-400 hover:bg-navy-800"
              }`}
            >
              Expected breaches only
            </button>
            <button
              type="button"
              aria-pressed={stressOnly}
              onClick={() => setStressOnly((v) => !v)}
              className={`rounded-md border px-2 py-1 text-[11px] ${
                stressOnly
                  ? "border-peri-400/40 bg-navy-800 text-peri-200"
                  : "border-[var(--color-line)] text-ink-400 hover:bg-navy-800"
              }`}
            >
              Stress breaches only
            </button>
          </>
        )}
        {s.priorAvailable && (
          <button
            type="button"
            aria-pressed={showPrior}
            onClick={() => setShowPrior((v) => !v)}
            className={`rounded-md border px-2 py-1 text-[11px] ${
              showPrior
                ? "border-peri-400/40 bg-navy-800 text-peri-200"
                : "border-[var(--color-line)] text-ink-400 hover:bg-navy-800"
            }`}
          >
            {showPrior ? "Showing prior period" : "Show prior period"}
          </button>
        )}
      </div>

      {/* Three-state comparison table */}
      <Card className="p-3" testId="concentration-table">
        <div
          className="grid grid-cols-[1.7fr_repeat(4,minmax(76px,1fr))_minmax(64px,0.8fr)_auto] items-end gap-2 border-b border-[var(--color-line-soft)] px-1 pb-2 text-[10px] uppercase tracking-wider text-ink-500"
          role="row"
        >
          <span>Test</span>
          <span className="text-right">Limit</span>
          <span className="text-right" aria-label="Funded, contractual position">
            {showPrior ? "Funded (prior)" : "Funded"}
            <span className="block text-[8px] normal-case text-ink-500">contractual</span>
          </span>
          <span className="text-right" aria-label="Expected Forecast, prediction">
            Expected Forecast
            <span className="block text-[8px] normal-case text-ink-500">prediction</span>
          </span>
          <span
            className="text-right text-ink-500/70"
            aria-label="Full Pipeline, stress, maximum exposure"
          >
            Full Pipeline
            <span className="block text-[8px] normal-case">stress — max exposure</span>
          </span>
          <span className="text-right">{statesAvailable ? "Move F→E" : "Δ period"}</span>
          <span>Risk</span>
        </div>
        {filtered.length === 0 && (
          <p className="px-1 py-3 text-[12px] text-ink-500" data-testid="concentration-no-match">
            No tests match the current filters.
          </p>
        )}
        {filtered.map((t) => {
          const chip = RISK_CHIP[categoryByTest.get(t.testId) ?? "ok"] ?? RISK_CHIP.ok;
          const expected = statePhrase(t.expected, t.unit);
          const full = statePhrase(t.fullPipeline, t.unit);
          const move = statesAvailable
            ? t.changeFundedToExpected
            : t.absoluteChange;
          return (
            <button
              type="button"
              key={t.testId}
              onClick={() => setSelectedId(selectedId === t.testId ? null : t.testId)}
              aria-expanded={selectedId === t.testId}
              aria-label={`${t.displayName} — ${STATUS_LABEL[t.status]}${
                t.expectedBreach ? ", breach expected" : ""
              }${t.fullPipelineBreach && !t.expectedBreach && t.status !== "breach"
                ? ", stress-only breach" : ""}`}
              className={`grid w-full grid-cols-[1.7fr_repeat(4,minmax(76px,1fr))_minmax(64px,0.8fr)_auto] items-center gap-2 border-b border-[var(--color-line-soft)] px-1 py-2 text-left text-[12px] last:border-0 hover:bg-navy-800/50 focus-visible:outline focus-visible:outline-1 focus-visible:outline-peri-400 ${
                selectedId === t.testId ? "bg-navy-800/60" : ""
              }`}
            >
              <span className="min-w-0">
                <span className="block truncate text-ink-200" title={t.displayName}>
                  {t.displayName}
                </span>
                <span className="block text-[10px] text-ink-500">
                  {categoryLabel(t.category)}
                  {t.dataStatus !== "ok" && " · data limited"}
                  {t.legacy && " · legacy source"}
                </span>
              </span>
              <span
                className="text-right font-mono tabular-nums text-ink-400"
                title={`Warning at ${(t.warningFraction * 100).toFixed(0)}% of limit`}
              >
                {t.threshold !== null
                  ? `${operatorGlyph(t.operator)} ${formatValue(t.threshold, t.unit)}`
                  : "—"}
              </span>
              <span className="text-right">
                <span className="block font-mono tabular-nums text-ink-200">
                  {formatValue(showPrior ? t.priorValue : t.currentValue, t.unit)}
                </span>
                <Badge tone={STATUS_TONE[t.status]}>
                  <span aria-hidden>{STATUS_GLYPH[t.status]}</span>
                  {STATUS_LABEL[t.status]}
                </Badge>
              </span>
              <span className="text-right" data-testid={`expected-${t.testId}`}>
                <span className="block font-mono tabular-nums text-ink-200">
                  {expected.value}
                </span>
                {t.expected && t.expected.status && t.expected.status !== "indicative_only" ? (
                  <Badge tone={STATUS_TONE[t.expected.status as ConcentrationTestStatus]}>
                    <span aria-hidden>
                      {STATUS_GLYPH[t.expected.status as ConcentrationTestStatus]}
                    </span>
                    {STATUS_LABEL[t.expected.status as ConcentrationTestStatus]}
                  </Badge>
                ) : (
                  <span className="text-[10px] text-ink-500">{expected.sub || "—"}</span>
                )}
                {t.expected && t.expected.status !== "indicative_only" && expected.sub && (
                  <span className="block text-[10px] text-ink-500">{expected.sub}</span>
                )}
              </span>
              <span className="text-right opacity-80" data-testid={`full-${t.testId}`}>
                <span className="block font-mono tabular-nums text-ink-300">
                  {full.value}
                </span>
                {t.fullPipeline?.status ? (
                  <span className="text-[10px] text-ink-500">
                    {STATUS_GLYPH[t.fullPipeline.status as ConcentrationTestStatus]}{" "}
                    {STATUS_LABEL[t.fullPipeline.status as ConcentrationTestStatus]}
                    {full.sub && ` · ${full.sub}`}
                  </span>
                ) : (
                  <span className="text-[10px] text-ink-500">{full.sub || "—"}</span>
                )}
              </span>
              <span
                className={`text-right font-mono tabular-nums ${
                  (move ?? 0) > 0 === (t.operator === "max") && move
                    ? "text-amber-300/90"
                    : "text-ink-400"
                }`}
              >
                {formatChange(move, t.unit)}
              </span>
              <span>
                <Badge tone={chip.tone}>{chip.label}</Badge>
              </span>
            </button>
          );
        })}
      </Card>

      {selected && (
        <ConcentrationDetailPanel
          client={client}
          portfolioId={portfolioId}
          portfolioContext={portfolioContext}
          test={selected}
          forecast={snapshot.forecast}
          statesAvailable={statesAvailable}
          fundedReportingDate={snapshot.reportingDate}
          statesReason={snapshot.states?.reason ?? null}
          onClose={() => setSelectedId(null)}
        />
      )}
    </div>
  );
}
