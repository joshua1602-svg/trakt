import { ArrowRight, Plus, TrendingUp } from "lucide-react";
import type { ForecastBridge } from "@/domain";
import { Badge } from "@/components/ui";
import { cn, formatGBP } from "@/lib/utils";

const READINESS_TONE: Record<ForecastBridge["forecastReadiness"]["status"], "mint" | "amber" | "rose"> = {
  ready: "mint",
  partial: "amber",
  blocked: "rose",
};

/**
 * The deterministic funded + pipeline forecast bridge:
 *
 *   current funded balance + probability-weighted expected pipeline
 *     = forecast funded balance
 *
 * All numbers are backend-derived (forecast_bridge.compute_forecast_bridge) —
 * this card only renders them.
 */
export function ForecastBridgeCard({ bridge }: { bridge: ForecastBridge | null }) {
  if (!bridge) return null;
  const readiness = bridge.forecastReadiness;
  const tone = READINESS_TONE[readiness.status] ?? "amber";

  return (
    <section className="rounded-xl border border-[var(--color-line)] bg-navy-900/40 p-5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2.5">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-navy-700/70 text-mint-400">
            <TrendingUp size={17} />
          </div>
          <h2 className="text-sm font-semibold text-ink-100">Funded + Pipeline Forecast</h2>
        </div>
        <Badge tone={tone}>Forecast {readiness.status}</Badge>
      </div>

      {/* funded + weighted pipeline = forecast (responsive equation) */}
      <div className="mt-4 grid grid-cols-1 items-stretch gap-2 sm:grid-cols-[1fr_auto_1fr_auto_1fr]">
        <BridgeTerm label="Current funded balance" value={formatGBP(bridge.fundedBalance)}
          hint={`${bridge.fundedLoanCount.toLocaleString("en-GB")} loans`} />
        <Operator icon={<Plus size={16} />} />
        <BridgeTerm label="Weighted expected pipeline" value={formatGBP(bridge.weightedExpectedFundedAmount)}
          hint={`${bridge.pipelineCaseCount.toLocaleString("en-GB")} pipeline cases`} />
        <Operator icon={<ArrowRight size={16} />} />
        <BridgeTerm label="Forecast funded balance" value={formatGBP(bridge.forecastFundedBalance)}
          hint={`${bridge.forecastLoanCount.toLocaleString("en-GB")} forecast loans`} emphasis />
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-ink-500">
        <span>
          Completion probability basis ·{" "}
          <span className="font-medium text-ink-300">{bridge.completionProbabilityBasis}</span>
        </span>
        {bridge.fundedReportingDate && (
          <span>Funded reporting date · {bridge.fundedReportingDate}</span>
        )}
        {bridge.pipelineAsOfDate && (
          <span>Pipeline as-of · {bridge.pipelineAsOfDate}</span>
        )}
      </div>

      {readiness.warnings.length > 0 && (
        <div className="mt-3 rounded-lg border border-amber-400/20 bg-amber-400/5 px-3 py-2 text-[11px] text-amber-300/90">
          {readiness.warnings.map((w, i) => (
            <div key={i}>⚠ {w}</div>
          ))}
        </div>
      )}
    </section>
  );
}

function BridgeTerm({
  label,
  value,
  hint,
  emphasis,
}: {
  label: string;
  value: string;
  hint?: string;
  emphasis?: boolean;
}) {
  return (
    <div
      className={cn(
        "rounded-lg border p-3.5",
        emphasis
          ? "border-mint-400/30 bg-mint-400/5"
          : "border-[var(--color-line-soft)] bg-navy-850/60",
      )}
    >
      <div className="text-[11px] font-medium uppercase tracking-wider text-ink-400">{label}</div>
      <div className={cn("mt-1.5 font-mono text-2xl font-semibold tabular-nums", emphasis ? "text-mint-300" : "text-ink-100")}>
        {value}
      </div>
      {hint && <div className="mt-1 text-[11px] text-ink-500">{hint}</div>}
    </div>
  );
}

function Operator({ icon }: { icon: React.ReactNode }) {
  return (
    <div className="flex items-center justify-center text-ink-500 sm:px-1" aria-hidden>
      {icon}
    </div>
  );
}
