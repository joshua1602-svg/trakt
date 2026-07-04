/**
 * TimingDisclosureBanner
 *
 * A lightweight, NON-BLOCKING note shown on pipeline / forecast views when the
 * latest pipeline weekly extract is dated later than the selected funded
 * reporting date. It discloses both anchors (funded actuals as-of, pipeline
 * extract as-of) instead of hiding or truncating the pipeline. Above the
 * configured lag threshold the tone strengthens from informational to a warning.
 *
 * Renders nothing when there is nothing to disclose (`level === "none"`).
 */
import { Clock, AlertTriangle } from "lucide-react";

import type { TimingDisclosure } from "@/domain/pipeline";

export function TimingDisclosureBanner({
  timing,
  className = "",
}: {
  timing?: TimingDisclosure | null;
  className?: string;
}) {
  if (!timing || timing.level === "none" || !timing.message) return null;
  const warning = timing.level === "warning";
  const tone = warning
    ? "border-amber-500/40 bg-amber-500/10 text-amber-200"
    : "border-peri-500/30 bg-peri-500/10 text-peri-100";
  const Icon = warning ? AlertTriangle : Clock;

  return (
    <div
      role="note"
      data-testid="pipeline-timing-disclosure"
      data-level={timing.level}
      className={`flex items-start gap-2 rounded-lg border px-3 py-2 text-[11px] leading-snug ${tone} ${className}`}
    >
      <Icon size={14} className="mt-0.5 shrink-0" />
      <div className="space-y-0.5">
        <p>{timing.message}</p>
        <p className="opacity-70">
          Funded actuals as of{" "}
          <span className="font-medium">{timing.fundedActualsAsOf ?? "—"}</span> · Pipeline
          extract as of{" "}
          <span className="font-medium">{timing.pipelineExtractAsOf ?? "—"}</span>
          {typeof timing.lagDays === "number" && timing.lagDays > 0
            ? ` · ${timing.lagDays}-day lag`
            : ""}
        </p>
      </div>
    </div>
  );
}
