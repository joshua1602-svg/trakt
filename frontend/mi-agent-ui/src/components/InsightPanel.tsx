import { useMemo } from "react";
import { Activity, Eye, Info, Lightbulb, Sparkles } from "lucide-react";
import type { Artifact } from "@/domain";
import { computeInsights, hasInsights, type Observation, type Severity } from "@/lib/insights";

const SEVERITY_META: Record<Severity, { icon: typeof Info; tone: string; label: string }> = {
  info: { icon: Info, tone: "text-ink-500", label: "Info" },
  watch: { icon: Eye, tone: "text-peri-300", label: "Watch" },
  significant: { icon: Activity, tone: "text-amber-300", label: "Significant" },
};

/**
 * Key Observations — the additive Insight Engine surface under a chart/table.
 * Everything is computed locally from the artifact rows; if computation throws
 * or yields nothing, the panel renders null and the result is unaffected.
 */
export function InsightPanel({
  artifact,
  onAsk,
}: {
  artifact: Artifact;
  onAsk?: (question: string) => void;
}) {
  const summary = useMemo(() => {
    try {
      return computeInsights(artifact, artifact.source.spec);
    } catch {
      return null; // never let insight analysis break the result
    }
  }, [artifact]);

  if (!hasInsights(summary)) return null;

  return (
    <div className="mt-3 rounded-lg border border-[var(--color-line-soft)] bg-navy-900/30 p-3">
      <div className="flex items-center gap-1.5">
        <Sparkles size={13} className="text-peri-300" />
        <span className="text-[11px] font-medium uppercase tracking-wider text-ink-400">Key observations</span>
      </div>

      <ul className="mt-2 space-y-1.5">
        {summary.observations.map((o) => (
          <ObservationRow key={o.id} observation={o} />
        ))}
      </ul>

      {summary.suggestions.length > 0 && (
        <div className="mt-3 border-t border-[var(--color-line-soft)] pt-2.5">
          <div className="mb-1.5 flex items-center gap-1.5">
            <Lightbulb size={12} className="text-ink-500" />
            <span className="text-[10px] font-medium uppercase tracking-wider text-ink-500">
              Suggested investigations
            </span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {summary.suggestions.map((s) => (
              <button
                key={`${s.kind}:${s.question}`}
                type="button"
                onClick={() => onAsk?.(s.question)}
                title={s.question}
                disabled={!onAsk}
                className="inline-flex items-center rounded-full border border-[var(--color-line)] bg-navy-800/50 px-2.5 py-1 text-[11px] text-ink-300 transition-colors enabled:hover:border-peri-400/40 enabled:hover:text-ink-100 disabled:opacity-60"
              >
                {s.label}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ObservationRow({ observation }: { observation: Observation }) {
  const meta = SEVERITY_META[observation.severity];
  const Icon = meta.icon;
  return (
    <li className="flex items-start gap-2 text-[12px] leading-relaxed text-ink-200">
      <Icon size={13} className={`mt-0.5 shrink-0 ${meta.tone}`} aria-label={meta.label} />
      <span>{observation.text}</span>
    </li>
  );
}
