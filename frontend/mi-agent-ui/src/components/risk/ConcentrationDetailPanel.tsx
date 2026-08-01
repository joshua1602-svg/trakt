/**
 * Detail panel for one concentration test: the approved definition and
 * parameters, current vs prior values, numerator / denominator, provenance and
 * approval history, trend across real governed snapshots, and the
 * contributing-loan drill-through (the evaluator's own population — the UI
 * never reconstructs the filter).
 */

import { useEffect, useState } from "react";
import { X } from "lucide-react";
import {
  Line,
  LineChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { AgentClient } from "@/api/AgentClient";
import type {
  ConcentrationDrillthrough,
  ConcentrationDrivers,
  ConcentrationHistory,
  ConcentrationStateBlock,
  ConcentrationTest,
  ConcentrationTestStatus,
  ForecastMethodology,
} from "@/domain";
import { Badge, Card } from "@/components/ui";
import { MethodologyBlock } from "./MethodologyBlock";
import {
  STATUS_LABEL,
  STATUS_GLYPH,
  STATUS_TONE,
  categoryLabel,
  formatChange,
  formatDate,
  formatValue,
  operatorGlyph,
} from "./concentrationShared";

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-[140px_1fr] items-baseline gap-2 py-1 text-[12px]">
      <dt className="text-ink-500">{label}</dt>
      <dd className="text-ink-300">{children}</dd>
    </div>
  );
}

function TrendChart({
  client,
  portfolioId,
  portfolioContext,
  test,
}: {
  client: AgentClient;
  portfolioId: string;
  portfolioContext?: string;
  test: ConcentrationTest;
}) {
  const [history, setHistory] = useState<ConcentrationHistory | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    client
      .getConcentrationHistory(portfolioId, test.testId, portfolioContext)
      .then((h) => {
        if (!cancelled) setHistory(h);
      })
      .catch(() => {
        /* keep prior */
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [client, portfolioId, portfolioContext, test.testId]);

  if (loading && !history) {
    return <p className="text-[12px] text-ink-500">Loading trend…</p>;
  }
  const series = history?.series?.[0];
  const points = (series?.points ?? []).filter((p) => p.value !== null);
  if (!history?.available || !series || points.length < 2) {
    return (
      <p className="text-[12px] text-ink-500" data-testid="concentration-trend-empty">
        {history?.reason ??
          "Not enough governed snapshot history for a trend — nothing is fabricated."}
      </p>
    );
  }
  const threshold = series.threshold;
  const warningLine =
    threshold !== null
      ? series.operator === "min"
        ? threshold / (series.warningFraction || 0.9)
        : threshold * (series.warningFraction || 0.9)
      : null;
  return (
    <div style={{ width: "100%", height: 200 }} data-testid="concentration-trend">
      <ResponsiveContainer>
        <LineChart data={points} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="#23304d" strokeDasharray="3 3" />
          <XAxis
            dataKey="reportingDate"
            tick={{ fill: "#8a97ad", fontSize: 11 }}
            tickFormatter={(d: string) => (d ? d.slice(0, 7) : "")}
          />
          <YAxis tick={{ fill: "#8a97ad", fontSize: 11 }} width={48} />
          <Tooltip
            contentStyle={{
              background: "#0f1626",
              border: "1px solid #23304d",
              fontSize: 12,
            }}
            formatter={(v: number) => [formatValue(v, series.unit), series.displayName]}
          />
          {threshold !== null && (
            <ReferenceLine
              y={threshold}
              stroke="#B23A48"
              strokeDasharray="4 4"
              label={{
                value: `limit ${operatorGlyph(series.operator)} ${formatValue(threshold, series.unit)}`,
                fill: "#e0607a",
                fontSize: 10,
                position: "insideTopRight",
              }}
            />
          )}
          {warningLine !== null && (
            <ReferenceLine y={warningLine} stroke="#E0A93B" strokeDasharray="2 4" />
          )}
          <Line
            type="monotone"
            dataKey="value"
            stroke="#919DD1"
            strokeWidth={2}
            dot={{ r: 3 }}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function DrillThrough({
  client,
  portfolioId,
  portfolioContext,
  test,
}: {
  client: AgentClient;
  portfolioId: string;
  portfolioContext?: string;
  test: ConcentrationTest;
}) {
  const [open, setOpen] = useState(false);
  const [drill, setDrill] = useState<ConcentrationDrillthrough | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!open || drill) return;
    let cancelled = false;
    setLoading(true);
    client
      .getConcentrationDrillthrough(portfolioId, test.testId, portfolioContext)
      .then((d) => {
        if (!cancelled) setDrill(d);
      })
      .catch(() => {
        /* keep prior */
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, drill, client, portfolioId, portfolioContext, test.testId]);

  return (
    <div data-testid="concentration-drillthrough">
      <button
        type="button"
        className="rounded-md border border-[var(--color-line)] px-2 py-1 text-[11px] text-peri-200 hover:bg-navy-800"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        {open ? "Hide contributing loans" : "Show contributing loans"}
      </button>
      {open && (
        <div className="mt-2">
          {loading && !drill && (
            <p className="text-[12px] text-ink-500">Loading contributing loans…</p>
          )}
          {drill && !drill.available && (
            <p className="text-[12px] text-ink-500">{drill.reason}</p>
          )}
          {drill?.available && (
            <>
              <p className="text-[11px] text-ink-500">
                {drill.loansInNumerator} loan(s) in the numerator
                {drill.numeratorValue != null && drill.denominatorValue != null && (
                  <>
                    {" "}— {formatValue(drill.numeratorValue, "currency")} of{" "}
                    {formatValue(drill.denominatorValue, "currency")} (
                    {(drill.denominatorBasis ?? "current_balance").replace(/_/g, " ")})
                  </>
                )}
                {drill.truncated ? ` · showing first ${drill.rows.length}` : ""}
                {drill.reconciles === false && " · does not reconcile — investigate"}
              </p>
              <div className="mt-1 max-h-64 overflow-auto rounded-lg border border-[var(--color-line-soft)]">
                <table className="w-full text-[11px]">
                  <thead>
                    <tr className="border-b border-[var(--color-line-soft)] text-left text-[10px] uppercase tracking-wider text-ink-500">
                      {drill.columns.map((c) => (
                        <th key={c} className="px-2 py-1 font-medium">
                          {c.replace(/_/g, " ")}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {drill.rows.map((row, i) => (
                      <tr key={i} className="border-b border-[var(--color-line-soft)] last:border-0">
                        {drill.columns.map((c) => (
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
            </>
          )}
        </div>
      )}
    </div>
  );
}

function StateCard({
  title,
  sublabel,
  block,
  test,
  muted,
  numerator,
  denominator,
}: {
  title: string;
  sublabel: string;
  block: ConcentrationStateBlock | null | undefined;
  test: ConcentrationTest;
  muted?: boolean;
  numerator?: number | null;
  denominator?: number | null;
}) {
  const unit = test.unit;
  return (
    <div
      className={`rounded-lg border border-[var(--color-line-soft)] p-2 ${
        muted ? "bg-navy-950/30 opacity-80" : "bg-navy-950/40"
      }`}
    >
      <p className="text-[10px] uppercase tracking-wider text-ink-500">
        {title} <span className="normal-case">({sublabel})</span>
      </p>
      {!block ? (
        <p className="text-[11px] text-ink-500">
          {test.forecastTreatment === "unsupported"
            ? "Not supported for this metric family."
            : "Not available."}
        </p>
      ) : block.status === "indicative_only" ? (
        <p className="text-[11px] text-ink-500">
          Indicative only — a fractional loan is never a maximum or a count. See Full
          Pipeline for the whole-loan stress view.
        </p>
      ) : (
        <>
          <p className="flex items-center gap-2">
            <span className="font-mono text-[16px] tabular-nums text-ink-100">
              {formatValue(block.value, unit)}
            </span>
            {block.status && (
              <Badge tone={STATUS_TONE[block.status as ConcentrationTestStatus]}>
                <span aria-hidden>
                  {STATUS_GLYPH[block.status as ConcentrationTestStatus]}
                </span>
                {STATUS_LABEL[block.status as ConcentrationTestStatus]}
              </Badge>
            )}
          </p>
          <p className="text-[10px] text-ink-500">
            {block.headroom !== null &&
              (block.headroom < 0
                ? `over by ${Math.abs(block.headroom).toFixed(2)}`
                : `headroom ${block.headroom.toFixed(2)}`) +
                (unit === "percent" ? "pp" : "")}
            {block.utilization !== null && ` · util. ${block.utilization.toFixed(1)}%`}
          </p>
          {numerator != null && denominator != null && (
            <p className="text-[10px] text-ink-500">
              N {formatValue(numerator, "currency")} / D{" "}
              {formatValue(denominator, "currency")}
            </p>
          )}
        </>
      )}
    </div>
  );
}

function DriversPanel({
  client,
  portfolioId,
  portfolioContext,
  test,
}: {
  client: AgentClient;
  portfolioId: string;
  portfolioContext?: string;
  test: ConcentrationTest;
}) {
  const [open, setOpen] = useState(false);
  const [drivers, setDrivers] = useState<ConcentrationDrivers | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!open || drivers) return;
    let cancelled = false;
    setLoading(true);
    client
      .getConcentrationDrivers(portfolioId, test.testId, portfolioContext)
      .then((d) => {
        if (!cancelled) setDrivers(d);
      })
      .catch(() => {
        /* keep prior */
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, drivers, client, portfolioId, portfolioContext, test.testId]);

  return (
    <div data-testid="pipeline-drivers">
      <button
        type="button"
        className="rounded-md border border-[var(--color-line)] px-2 py-1 text-[11px] text-peri-200 hover:bg-navy-800"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        {open ? "Hide pipeline drivers" : "Show pipeline drivers"}
      </button>
      {open && (
        <div className="mt-2">
          {loading && !drivers && (
            <p className="text-[12px] text-ink-500">Loading pipeline drivers…</p>
          )}
          {drivers && !drivers.available && (
            <p className="text-[12px] text-ink-500">{drivers.reason}</p>
          )}
          {drivers?.available && (
            <>
              <p className="text-[11px] text-ink-500">
                Expected numerator movement{" "}
                {formatValue(drivers.expectedNumeratorMovement ?? null, "currency")} from{" "}
                {drivers.driverCount} pipeline case(s)
                {drivers.topShareOfMovement != null &&
                  ` · listed cases drive ${(drivers.topShareOfMovement * 100).toFixed(0)}%`}
                {drivers.reconciles === false && (
                  <span className="text-amber-300/90">
                    {" "}
                    · does not reconcile — investigate
                  </span>
                )}
                {drivers.reconciles !== false &&
                  " · contributions reconcile to the Expected Forecast numerator"}
              </p>
              <div className="mt-1 max-h-64 overflow-auto rounded-lg border border-[var(--color-line-soft)]">
                <table className="w-full text-[11px]">
                  <thead>
                    <tr className="border-b border-[var(--color-line-soft)] text-left text-[10px] uppercase tracking-wider text-ink-500">
                      <th className="px-2 py-1 font-medium">Case</th>
                      <th className="px-2 py-1 text-right font-medium">Balance</th>
                      <th className="px-2 py-1 font-medium">Stage</th>
                      <th className="px-2 py-1 text-right font-medium">Prob.</th>
                      <th className="px-2 py-1 text-right font-medium">Expected</th>
                      <th className="px-2 py-1 font-medium">Exp. month</th>
                      <th className="px-2 py-1 font-medium">Impact</th>
                    </tr>
                  </thead>
                  <tbody>
                    {drivers.drivers.map((d) => (
                      <tr
                        key={d.caseId}
                        className="border-b border-[var(--color-line-soft)] last:border-0"
                      >
                        <td className="px-2 py-1 font-mono text-ink-300">{d.caseId}</td>
                        <td className="px-2 py-1 text-right font-mono tabular-nums text-ink-300">
                          {d.balance.toLocaleString("en-GB")}
                        </td>
                        <td className="px-2 py-1 text-ink-400">{d.stage ?? "—"}</td>
                        <td
                          className="px-2 py-1 text-right font-mono tabular-nums text-ink-300"
                          title={d.probabilitySource ?? undefined}
                        >
                          {d.completionProbability.toFixed(2)}
                        </td>
                        <td className="px-2 py-1 text-right font-mono tabular-nums text-ink-300">
                          {d.expectedContribution.toLocaleString("en-GB")}
                        </td>
                        <td className="px-2 py-1 text-ink-400">
                          {d.expectedCompletionMonth ?? "—"}
                        </td>
                        <td className="px-2 py-1 text-amber-300/90">
                          {d.impact === "tips_breach"
                            ? "tips breach"
                            : d.impact === "tips_warning"
                              ? "tips warning"
                              : ""}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export function ConcentrationDetailPanel({
  client,
  portfolioId,
  portfolioContext,
  test,
  forecast,
  statesAvailable,
  onClose,
}: {
  client: AgentClient;
  portfolioId: string;
  portfolioContext?: string;
  test: ConcentrationTest;
  forecast?: ForecastMethodology;
  statesAvailable?: boolean;
  onClose: () => void;
}) {
  const [methodologyOpen, setMethodologyOpen] = useState(false);
  const params = Object.entries(test.parameters ?? {});
  return (
    <Card className="p-4" testId="concentration-detail">
      <div className="flex items-start justify-between gap-2">
        <div>
          <h3 className="text-[14px] font-semibold text-ink-100">{test.displayName}</h3>
          <p className="text-[11px] text-ink-500">
            {categoryLabel(test.category)} · {test.severity} severity
            {test.effectiveDate && ` · effective ${formatDate(test.effectiveDate)}`}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge tone={STATUS_TONE[test.status]}>
            <span aria-hidden>{STATUS_GLYPH[test.status]}</span>
            {STATUS_LABEL[test.status]}
          </Badge>
          <button
            type="button"
            aria-label="Close test detail"
            className="rounded-md p-1 text-ink-500 hover:bg-navy-800 hover:text-ink-300"
            onClick={onClose}
          >
            <X size={14} />
          </button>
        </div>
      </div>

      {statesAvailable && !test.legacy && (
        <div
          className="mt-3 grid gap-2 md:grid-cols-3"
          data-testid="state-comparison"
        >
          <StateCard
            title="Funded"
            sublabel="contractual"
            block={{
              value: test.currentValue,
              status: test.status,
              headroom: test.headroom,
              utilization: test.utilization,
              breachAmount: test.breachAmount,
            }}
            test={test}
            numerator={test.numeratorValue}
            denominator={test.denominatorValue}
          />
          <StateCard
            title="Expected Forecast"
            sublabel="prediction"
            block={test.expected}
            test={test}
            numerator={test.expectedNumerator}
            denominator={test.expectedDenominator}
          />
          <StateCard
            title="Full Pipeline"
            sublabel="stress — max exposure"
            block={test.fullPipeline}
            test={test}
            muted
            numerator={test.fullPipelineNumerator}
            denominator={test.fullPipelineDenominator}
          />
        </div>
      )}
      {test.expectedBreach && test.expectedBreachHorizon?.period && (
        <p className="mt-1 text-[11px] text-amber-300/90" data-testid="breach-horizon">
          Expected breach horizon: {test.expectedBreachHorizon.period} (cumulative
          expected completions cross the limit).
        </p>
      )}
      {test.forecastTreatmentNote &&
        test.forecastTreatment !== "supported_with_exposure_weighting" &&
        statesAvailable && (
          <p className="mt-1 text-[11px] text-ink-500">{test.forecastTreatmentNote}</p>
        )}

      <div className="mt-3 grid gap-x-8 gap-y-1 md:grid-cols-2">
        <dl>
          <Row label="Current value">
            <span className="font-mono tabular-nums text-ink-100">
              {formatValue(test.currentValue, test.unit)}
            </span>
            {test.status === "unavailable" || test.status === "insufficient_data" ? (
              <span className="ml-2 text-ink-500">({test.notes || test.dataStatus})</span>
            ) : null}
          </Row>
          <Row label="Approved limit">
            {test.threshold !== null ? (
              <>
                {operatorGlyph(test.operator)} {formatValue(test.threshold, test.unit)}
                <span className="ml-1 text-ink-500">
                  (warning at {(test.warningFraction * 100).toFixed(0)}% of limit)
                </span>
              </>
            ) : (
              "—"
            )}
          </Row>
          <Row label="Headroom">
            <span className="font-mono tabular-nums">
              {formatValue(test.headroom, test.unit === "percent" ? "percent" : test.unit)}
            </span>
            {test.utilization !== null && (
              <span className="ml-2 text-ink-500">
                utilization {test.utilization.toFixed(1)}%
              </span>
            )}
          </Row>
          <Row label="Period movement">
            {formatChange(test.absoluteChange, test.unit)}
            {test.priorValue !== null && (
              <span className="ml-1 text-ink-500">
                (prior {formatValue(test.priorValue, test.unit)} at{" "}
                {formatDate(test.priorReportingDate)})
              </span>
            )}
            {test.statusTransition && (
              <span className="ml-2 text-amber-300/90">
                {test.statusTransition.replace("->", "→")}
              </span>
            )}
            {!test.priorAvailable && (
              <span className="text-ink-500">No prior governed snapshot.</span>
            )}
          </Row>
          <Row label="Numerator">
            {test.numeratorValue !== null
              ? `${formatValue(test.numeratorValue, "currency")} · ${test.loansInNumerator} loan(s)`
              : "—"}
          </Row>
          <Row label="Denominator">
            {test.denominatorValue !== null
              ? `${formatValue(test.denominatorValue, "currency")} (${(
                  test.denominatorBasis || "current_balance"
                ).replace(/_/g, " ")}) · ${test.totalLoans} loan(s)`
              : "—"}
          </Row>
        </dl>
        <dl>
          <Row label="Definition">
            {test.definition.description || "—"}
            {test.definition.numerator && (
              <span className="block text-ink-500">{test.definition.numerator}</span>
            )}
          </Row>
          {params.length > 0 && (
            <Row label="Approved parameters">
              <span className="font-mono text-[11px]">
                {params.map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(" · ")}
              </span>
            </Row>
          )}
          {test.definition.definitionNotes && (
            <Row label="Confirmed at onboarding">{test.definition.definitionNotes}</Row>
          )}
          {test.provenance.sourceText && (
            <Row label="Source wording">
              <q className="text-ink-400">{test.provenance.sourceText}</q>
              {test.provenance.locator && (
                <span className="ml-1 text-ink-500">({test.provenance.locator})</span>
              )}
            </Row>
          )}
          <Row label="Approval">
            {test.provenance.approvedBy ? (
              <>
                {test.provenance.approvedBy} on {formatDate(test.provenance.approvedAt)}
                {test.provenance.approvalComments && (
                  <span className="block text-ink-500">
                    “{test.provenance.approvalComments}”
                  </span>
                )}
              </>
            ) : (
              <span className="text-amber-300/90">
                Not operator-approved (legacy extracted limit).
              </span>
            )}
          </Row>
          <Row label="Configuration">
            {test.configurationVersion !== null
              ? `version ${test.configurationVersion}`
              : "—"}
            {test.evaluatedAt && (
              <span className="ml-1 text-ink-500">
                · evaluated {formatDate(test.evaluatedAt)}
              </span>
            )}
          </Row>
          {test.missingFields.length > 0 && (
            <Row label="Missing fields">
              <span className="font-mono text-[11px]">{test.missingFields.join(", ")}</span>
            </Row>
          )}
        </dl>
      </div>

      <div className="mt-4 space-y-3">
        <TrendChart
          client={client}
          portfolioId={portfolioId}
          portfolioContext={portfolioContext}
          test={test}
        />
        {statesAvailable && !test.legacy && (
          <DriversPanel
            client={client}
            portfolioId={portfolioId}
            portfolioContext={portfolioContext}
            test={test}
          />
        )}
        {forecast?.available && (
          <div>
            <button
              type="button"
              className="rounded-md border border-[var(--color-line)] px-2 py-1 text-[11px] text-peri-200 hover:bg-navy-800"
              aria-expanded={methodologyOpen}
              onClick={() => setMethodologyOpen((v) => !v)}
            >
              {methodologyOpen ? "Hide forecast methodology" : "Forecast methodology"}
            </button>
            {methodologyOpen && (
              <div className="mt-2">
                <MethodologyBlock forecast={forecast} />
              </div>
            )}
          </div>
        )}
        {!test.legacy && (
          <DrillThrough
            client={client}
            portfolioId={portfolioId}
            portfolioContext={portfolioContext}
            test={test}
          />
        )}
      </div>
    </Card>
  );
}
