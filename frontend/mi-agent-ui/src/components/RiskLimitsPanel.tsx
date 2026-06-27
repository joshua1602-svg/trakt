import { useEffect, useState } from "react";
import { ShieldAlert, ArrowUpRight, ArrowDownRight, Minus } from "lucide-react";
import type { AgentClient } from "@/api";
import type { RiskLimitsSnapshot, RiskLimitTest, RiskStatus } from "@/domain";
import { Badge, Card } from "@/components/ui";

const STATUS_TONE: Record<RiskStatus, "mint" | "amber" | "rose" | "navy" | "neutral"> = {
  green: "mint",
  amber: "amber",
  red: "rose",
  needs_review: "navy",
  unavailable: "neutral",
};
const STATUS_LABEL: Record<RiskStatus, string> = {
  green: "Pass",
  amber: "Warn",
  red: "Breach",
  needs_review: "Needs review",
  unavailable: "Unavailable",
};

const CATEGORY_TITLES: Record<string, string> = {
  geographic_concentration: "Geographic concentration",
  broker_concentration: "Broker / intermediary concentration",
  large_loan_concentration: "Loan size concentration",
  ltv_limit: "LTV / valuation",
  interest_rate_limit: "Interest rate / WAC",
  borrower_concentration: "Borrower concentration",
  joint_borrower_limit: "Joint borrowers",
  age_limit: "Borrower age",
  property_value_concentration: "Property value",
  other: "Other tests",
};

function pct(v: number | null, unit: string | null): string {
  if (v == null) return "—";
  if (unit === "count") return v.toLocaleString("en-GB");
  if (unit === "gbp") return `£${Math.round(v).toLocaleString("en-GB")}`;
  return `${v.toFixed(1)}%`;
}

function movementIcon(m: number | null) {
  if (m == null || m === 0) return <Minus size={12} className="text-ink-500" />;
  return m > 0
    ? <ArrowUpRight size={12} className="text-rose-300" />
    : <ArrowDownRight size={12} className="text-mint-300" />;
}

function SummaryCard({ label, value, tone }: { label: string; value: string | number; tone?: string }) {
  return (
    <div className="rounded-lg border border-[var(--color-line-soft)] bg-navy-900/50 p-3">
      <div className="text-[10px] uppercase tracking-wider text-ink-500">{label}</div>
      <div className={`mt-1 text-lg font-semibold ${tone ?? "text-ink-100"}`}>{value}</div>
    </div>
  );
}

function TestRow({ test }: { test: RiskLimitTest }) {
  return (
    <div className="grid grid-cols-[1.6fr_repeat(4,1fr)_auto] items-center gap-2 border-b border-[var(--color-line-soft)] px-1 py-2 text-[12px] last:border-0">
      <div className="min-w-0">
        <div className="truncate text-ink-200" title={test.sourceSnippet ?? undefined}>{test.label}</div>
        <div className="truncate text-[10px] text-ink-500">{test.actualBasis}</div>
      </div>
      <div className="text-ink-100">{pct(test.actualValue, test.unit)}</div>
      <div className="text-ink-400">{pct(test.limitValue, test.unit)}</div>
      <div className={test.headroom != null && test.headroom < 0 ? "text-rose-300" : "text-ink-300"}>
        {test.headroom == null ? "—" : `${test.headroom.toFixed(1)}`}
      </div>
      <div className="flex items-center gap-1 text-ink-400">
        {movementIcon(test.movementVsPrior)}
        <span>{test.movementVsPrior == null ? "—" : `${test.movementVsPrior > 0 ? "+" : ""}${test.movementVsPrior.toFixed(1)}`}</span>
      </div>
      <Badge tone={STATUS_TONE[test.status]}>{STATUS_LABEL[test.status]}</Badge>
    </div>
  );
}

function CategoryTable({ title, tests }: { title: string; tests: RiskLimitTest[] }) {
  if (!tests.length) return null;
  return (
    <Card className="p-4">
      <div className="mb-2 text-[12px] font-semibold text-ink-200">{title}</div>
      <div className="grid grid-cols-[1.6fr_repeat(4,1fr)_auto] gap-2 px-1 pb-1 text-[10px] uppercase tracking-wider text-ink-500">
        <span>Test</span><span>Actual</span><span>Limit</span><span>Headroom</span><span>Movement</span><span>Status</span>
      </div>
      {tests.map((t) => <TestRow key={t.limitId} test={t} />)}
      {tests.some((t) => t.missingFields.length > 0) && (
        <p className="mt-2 text-[10px] text-ink-500">
          Unavailable tests list the missing field(s):{" "}
          {Array.from(new Set(tests.flatMap((t) => t.missingFields))).join(", ")}.
        </p>
      )}
    </Card>
  );
}

/**
 * Risk Limits / Concentration panel — Schedule 8 extracted limits vs funded
 * actual exposure: summary cards, geographic + broker concentration, other
 * concentration tests, key observations and lineage. Controlled
 * unavailable / needs-review states are surfaced, never hidden.
 */
export function RiskLimitsPanel({
  client, portfolioId,
}: {
  client: AgentClient;
  portfolioId: string;
}) {
  const [data, setData] = useState<RiskLimitsSnapshot | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    client
      .getRiskLimits(portfolioId)
      .then((d) => { if (!cancelled) setData(d); })
      .catch(() => { /* keep prior */ })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [client, portfolioId]);

  return (
    <section className="space-y-4" data-testid="risk-limits-panel">
      <div className="flex items-center gap-2 text-sm font-semibold text-ink-100">
        <ShieldAlert size={16} className="text-peri-300" /> Risk Limits / Concentration
      </div>

      {loading && !data && <p className="text-[12px] text-ink-500">Loading risk limits…</p>}

      {data && !data.available && (
        <Card className="p-4">
          <div className="text-[13px] font-medium text-amber-300">Limits unavailable — extraction required</div>
          <p className="mt-1 text-[12px] text-ink-400">
            {data.limitsReason ?? "No Schedule 8 limits are available for this portfolio."}
            {" "}Observed concentrations are still shown where data exists.
          </p>
        </Card>
      )}

      {data && (
        <>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
            <SummaryCard label="Tests passed" value={data.summary.testsPassed} tone="text-mint-300" />
            <SummaryCard label="Warnings" value={data.summary.warnings} tone="text-amber-300" />
            <SummaryCard label="Breaches" value={data.summary.breaches} tone="text-rose-300" />
            <SummaryCard label="Needs review" value={data.summary.needsReview} tone="text-peri-300" />
            <SummaryCard
              label="Closest headroom"
              value={data.summary.closestHeadroom ? `${data.summary.closestHeadroom.headroom.toFixed(1)} pp` : "—"}
            />
            <SummaryCard
              label="Largest concentration"
              value={data.summary.largestConcentration ? `${data.summary.largestConcentration.actualValue.toFixed(1)}%` : "—"}
            />
          </div>

          {Object.entries(data.testsByCategory).map(([cat, tests]) => (
            <CategoryTable key={cat} title={CATEGORY_TITLES[cat] ?? cat} tests={tests} />
          ))}

          {data.observations.length > 0 && (
            <Card className="p-4">
              <div className="mb-2 text-[12px] font-semibold text-ink-200">Key observations</div>
              <ul className="space-y-1 text-[12px] text-ink-300">
                {data.observations.map((o, i) => <li key={i}>• {o}</li>)}
              </ul>
            </Card>
          )}

          <Card className="p-4">
            <div className="mb-2 text-[12px] font-semibold text-ink-200">Lineage</div>
            <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-[11px] sm:grid-cols-3">
              <div><dt className="text-ink-500">Limit source</dt><dd className="text-ink-300">{data.limitsSource}</dd></div>
              <div><dt className="text-ink-500">Source document</dt><dd className="text-ink-300">{data.lineage?.sourceDocument ?? "—"}</dd></div>
              <div><dt className="text-ink-500">Exposure basis</dt><dd className="text-ink-300">{data.lineage?.exposureBasis ?? "funded"}</dd></div>
              <div><dt className="text-ink-500">Data source</dt><dd className="text-ink-300">{data.lineage?.dataSource ?? "—"}</dd></div>
              <div><dt className="text-ink-500">Reporting date</dt><dd className="text-ink-300">{data.reportingDate ?? "—"}</dd></div>
              <div><dt className="text-ink-500">Limits status</dt><dd className="text-ink-300">{data.limitsStatus}</dd></div>
            </dl>
          </Card>
        </>
      )}
    </section>
  );
}
