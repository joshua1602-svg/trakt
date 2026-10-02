/**
 * Forecast methodology disclosure — renders the governed methodology payload
 * verbatim (wireframe 05). The UI adds no interpretation: every line maps to
 * a field of the service's forecast block, and MI Query answers "how was the
 * expected forecast calculated?" from the same payload.
 */

import type { ForecastMethodology } from "@/domain";
import { formatDate } from "./concentrationShared";

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-[110px_1fr] items-baseline gap-2 py-0.5 text-[11px]">
      <dt className="text-ink-500">{label}</dt>
      <dd className="text-ink-300">{children}</dd>
    </div>
  );
}

const pct = (v: number | null | undefined) => (v == null ? "n/a" : `${(v * 100).toFixed(0)}%`);
const title = (s: string) => `${s.charAt(0)}${s.slice(1).toLowerCase()}`;

/** The stage run-off forecast — what the expected state actually weights by. */
function RunoffBlock({ forecast }: { forecast: ForecastMethodology }) {
  const runoff = forecast.runoff!;
  const windows = Object.entries(runoff.stages ?? {})
    .filter(([, s]) => s.windowDays != null)
    .map(([st, s]) => `${title(st)} ${s.windowDays}d${s.windowBasis === "fallback" ? " (configured)" : ""}`);
  const nf = forecast.notForecast ?? {};
  return (
    <dl data-testid="methodology-block" className="rounded-lg border border-[var(--color-line-soft)] bg-navy-950/40 p-2">
      <Row label="Model">
        Stage run-off measured from the client's weekly pipeline extracts. No machine
        learning; no invented probabilities.
      </Row>
      <Row label="Window">
        {formatDate(forecast.observationWindowStart)} →{" "}
        {formatDate(forecast.observationWindowEnd)} · {forecast.weeklyExtractsUsed}{" "}
        weekly extracts
      </Row>
      <Row label="Pull-through">
        Application → Offer {pct(runoff.appToOfferPullThrough)} · Offer → Completion{" "}
        {pct(runoff.offerToCompletionPullThrough)}. A live case's chance of completing
        falls with the time it has already spent in its stage.
      </Row>
      {windows.length > 0 && (
        <Row label="Validity">
          {windows.join(" · ")}. A case past its stage window is treated as lapsed.
        </Row>
      )}
      <Row label="Not forecast">
        {nf.kfiCount ?? 0} KFIs (top of funnel) · {nf.lapsedCount ?? 0} past their stage
        window. Withdrawn and completed cases are never counted.
      </Row>
      <Row label="Full Pipeline">
        Every live Application and Offer at 100%, ignoring probabilities. A
        maximum-exposure stress, not a prediction.
      </Row>
      {forecast.currentSnapshot && (
        <Row label="Sources">{forecast.currentSnapshot} (current snapshot)</Row>
      )}
    </dl>
  );
}

export function MethodologyBlock({ forecast }: { forecast: ForecastMethodology }) {
  if (forecast.runoff?.available) return <RunoffBlock forecast={forecast} />;
  const rates = forecast.stageRates ?? {};
  const timing = forecast.stageTiming ?? {};
  const stageBits = ["KFI", "APPLICATION", "OFFER"]
    .filter((s) => rates[s]?.rate != null)
    .map((s) => {
      const r = rates[s]!;
      return `${s.charAt(0)}${s.slice(1).toLowerCase()} ${r.rate!.toFixed(2)} (observed ${
        r.observed
      }${r.sufficient ? " · sufficient" : " · below floor — configured fallback"})`;
    });
  const timingBits = Object.entries(timing).map(
    ([s, t]) => `${s.charAt(0)}${s.slice(1).toLowerCase()} ${t.medianDays}d`,
  );
  const excl = forecast.excludedStageCounts ?? {};
  return (
    <dl data-testid="methodology-block" className="rounded-lg border border-[var(--color-line-soft)] bg-navy-950/40 p-2">
      <Row label="Model">
        Deterministic completion-trend model over the client's weekly pipeline
        extracts. No machine learning; no invented probabilities.
      </Row>
      <Row label="Window">
        {formatDate(forecast.observationWindowStart)} →{" "}
        {formatDate(forecast.observationWindowEnd)} · {forecast.weeklyExtractsUsed}{" "}
        weekly extracts · {forecast.trackedCaseCount} cases tracked ·{" "}
        {forecast.observedCompletionCount} observed completions
      </Row>
      {stageBits.length > 0 && (
        <Row label="Stage rates">
          {stageBits.join(" · ")}. A stage needs ≥ {forecast.minObservations} observed
          cases before its empirical rate is trusted; otherwise the configured stage
          assumption applies.
        </Row>
      )}
      {forecast.basis && <Row label="Basis">{forecast.basis.replace(/_/g, " ")}</Row>}
      {timingBits.length > 0 && (
        <Row label="Timing">
          Median observed days to completion: {timingBits.join(" · ")}. Expected
          completion months derive from explicit dates where supplied, else stage
          timing offsets.
        </Row>
      )}
      {Object.keys(excl).length > 0 && (
        <Row label="Exclusions">
          Withdrawn / cancelled / declined / lapsed cases are never counted or weighted
          ({Object.entries(excl)
            .map(([k, v]) => `${k.toLowerCase()}: ${v}`)
            .join(", ")}
          ).
        </Row>
      )}
      <Row label="Full Pipeline">
        Ignores probabilities entirely: every active in-scope case at 100%. A
        maximum-exposure stress, not a prediction.
      </Row>
      {forecast.pointInTimeNote && (
        <Row label="Limitations">{forecast.pointInTimeNote}</Row>
      )}
      {forecast.currentSnapshot && (
        <Row label="Sources">{forecast.currentSnapshot} (current snapshot)</Row>
      )}
    </dl>
  );
}
