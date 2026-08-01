/**
 * Funded → Risk Limits — the approved concentration tests against the funded
 * book, from the ONE governed evaluation service (`/mi/concentration-tests`).
 *
 * The workspace renders governed results verbatim: it never recalculates a
 * test, never treats an unavailable value as zero, and always shows where a
 * limit came from (approved configuration version, or the legacy extracted
 * source explicitly marked unapproved). Status is always conveyed by label
 * and glyph as well as colour.
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
import {
  STATUS_GLYPH,
  STATUS_LABEL,
  STATUS_TONE,
  categoryLabel,
  formatChange,
  formatDate,
  formatValue,
  operatorGlyph,
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

type SortKey = "status" | "name" | "headroom" | "change";

function SummaryTile({
  label,
  value,
  tone,
  testId,
}: {
  label: string;
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
  const [sortKey, setSortKey] = useState<SortKey>("status");
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
  const categories = useMemo(
    () => Array.from(new Set(tests.map((t) => t.category))).sort(),
    [tests],
  );

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    const rows = tests.filter((t) => {
      if (category !== "all" && t.category !== category) return false;
      if (status !== "all") {
        if (status === "unavailable") {
          if (!["unavailable", "insufficient_data"].includes(t.status)) return false;
        } else if (t.status !== status) return false;
      }
      if (q && !t.displayName.toLowerCase().includes(q)) return false;
      return true;
    });
    const sorted = [...rows];
    sorted.sort((a, b) => {
      if (sortKey === "name") return a.displayName.localeCompare(b.displayName);
      if (sortKey === "headroom") {
        const ah = a.headroom ?? Number.POSITIVE_INFINITY;
        const bh = b.headroom ?? Number.POSITIVE_INFINITY;
        return ah - bh;
      }
      if (sortKey === "change") {
        return Math.abs(b.absoluteChange ?? 0) - Math.abs(a.absoluteChange ?? 0);
      }
      return (
        STATUS_ORDER[a.status] - STATUS_ORDER[b.status] ||
        (a.headroom ?? Number.POSITIVE_INFINITY) - (b.headroom ?? Number.POSITIVE_INFINITY)
      );
    });
    return sorted;
  }, [tests, search, category, status, sortKey]);

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
          {!snapshot.fundedDataAvailable && (
            <span className="ml-1 text-amber-300/90">· funded data unavailable</span>
          )}
        </p>
      </div>

      <SourceBanner snapshot={snapshot} />

      {/* Portfolio summary */}
      <div
        className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6"
        data-testid="concentration-summary"
      >
        <SummaryTile
          label="Overall"
          value={STATUS_LABEL[s.overallStatus]}
          tone={
            s.overallStatus === "breach"
              ? "rose"
              : s.overallStatus === "warning"
                ? "amber"
                : s.overallStatus === "pass"
                  ? "mint"
                  : "neutral"
          }
          testId="concentration-overall"
        />
        <SummaryTile label="Active tests" value={s.activeTests} />
        <SummaryTile
          label="Breaches"
          value={s.breaches}
          tone={s.breaches ? "rose" : "neutral"}
          testId="concentration-breaches"
        />
        <SummaryTile
          label="Warnings"
          value={s.warnings}
          tone={s.warnings ? "amber" : "neutral"}
        />
        <SummaryTile label="Passes" value={s.passes} tone={s.passes ? "mint" : "neutral"} />
        <SummaryTile
          label="Deteriorated"
          value={s.priorAvailable ? s.deteriorations : "—"}
          tone={s.deteriorations ? "amber" : "neutral"}
          testId="concentration-deteriorations"
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
          value={sortKey}
          onChange={(e) => setSortKey(e.target.value as SortKey)}
          className="rounded-md border border-[var(--color-line)] bg-navy-900/70 px-2 py-1 text-[12px] text-ink-300"
        >
          <option value="status">Sort: severity</option>
          <option value="headroom">Sort: headroom</option>
          <option value="change">Sort: movement</option>
          <option value="name">Sort: name</option>
        </select>
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

      {/* Primary limits table */}
      <Card className="p-3" testId="concentration-table">
        <div
          className="grid grid-cols-[1.8fr_repeat(4,minmax(72px,1fr))_auto] items-center gap-2 border-b border-[var(--color-line-soft)] px-1 pb-2 text-[10px] uppercase tracking-wider text-ink-500"
          role="row"
        >
          <span>Test</span>
          <span className="text-right">{showPrior ? "Prior" : "Current"}</span>
          <span className="text-right">Limit</span>
          <span className="text-right">Headroom</span>
          <span className="text-right">Δ period</span>
          <span>Status</span>
        </div>
        {filtered.length === 0 && (
          <p className="px-1 py-3 text-[12px] text-ink-500" data-testid="concentration-no-match">
            No tests match the current filters.
          </p>
        )}
        {filtered.map((t) => (
          <button
            type="button"
            key={t.testId}
            onClick={() => setSelectedId(selectedId === t.testId ? null : t.testId)}
            aria-expanded={selectedId === t.testId}
            aria-label={`${t.displayName} — ${STATUS_LABEL[t.status]}`}
            className={`grid w-full grid-cols-[1.8fr_repeat(4,minmax(72px,1fr))_auto] items-center gap-2 border-b border-[var(--color-line-soft)] px-1 py-2 text-left text-[12px] last:border-0 hover:bg-navy-800/50 focus-visible:outline focus-visible:outline-1 focus-visible:outline-peri-400 ${
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
            <span className="text-right font-mono tabular-nums text-ink-200">
              {formatValue(showPrior ? t.priorValue : t.currentValue, t.unit)}
            </span>
            <span
              className="text-right font-mono tabular-nums text-ink-400"
              title={`Warning at ${(t.warningFraction * 100).toFixed(0)}% of limit`}
            >
              {t.threshold !== null
                ? `${operatorGlyph(t.operator)} ${formatValue(t.threshold, t.unit)}`
                : "—"}
            </span>
            <span className="text-right font-mono tabular-nums text-ink-400">
              {formatValue(t.headroom, t.unit)}
            </span>
            <span
              className={`text-right font-mono tabular-nums ${
                t.deteriorated ? "text-amber-300/90" : "text-ink-400"
              }`}
            >
              {formatChange(t.absoluteChange, t.unit)}
            </span>
            <span className="flex items-center gap-1">
              {t.priorStatus && t.priorStatus !== t.status && (
                <>
                  <Badge tone={STATUS_TONE[t.priorStatus]} className="opacity-60">
                    {STATUS_LABEL[t.priorStatus]}
                  </Badge>
                  <span aria-hidden className="text-ink-500">
                    →
                  </span>
                </>
              )}
              <Badge tone={STATUS_TONE[t.status]}>
                <span aria-hidden>{STATUS_GLYPH[t.status]}</span>
                {STATUS_LABEL[t.status]}
              </Badge>
            </span>
          </button>
        ))}
      </Card>

      {selected && (
        <ConcentrationDetailPanel
          client={client}
          portfolioId={portfolioId}
          portfolioContext={portfolioContext}
          test={selected}
          onClose={() => setSelectedId(null)}
        />
      )}
    </div>
  );
}
