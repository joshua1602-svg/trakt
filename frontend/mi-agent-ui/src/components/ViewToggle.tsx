import { GitBranch, Landmark, ShieldAlert, TrendingUp } from "lucide-react";
import type { WorkspaceView } from "@/domain";
import { cn } from "@/lib/utils";

// Top-level tabs follow the book lifecycle: Funded → Pipeline → Forecast, plus
// the cross-cutting Risk Limits monitor. Each of Funded/Pipeline/Forecast hosts
// its own sub-tabs (stratifications / geography / evolution / cohorts / …).
const VIEWS: { id: WorkspaceView; label: string; icon: typeof Landmark }[] = [
  { id: "funded", label: "Funded", icon: Landmark },
  { id: "pipeline", label: "Pipeline", icon: GitBranch },
  { id: "forecast", label: "Forecast", icon: TrendingUp },
  { id: "risk_limits", label: "Risk Limits", icon: ShieldAlert },
];

/** Segmented control selecting the active MI workspace view.
 *
 * ``disabledViews`` disables views that don't apply to the active scope — e.g.
 * an acquired-only source portfolio has no origination Pipeline / Forecast, so
 * those tabs are disabled. The source-portfolio lens and this view toggle are
 * independent axes. */
export function ViewToggle({
  active,
  onChange,
  disabledViews = [],
  className,
}: {
  active: WorkspaceView;
  onChange: (view: WorkspaceView) => void;
  disabledViews?: WorkspaceView[];
  /** Optional wrapper classes (e.g. `flex-1` to span the dashboard width). */
  className?: string;
}) {
  const disabled = new Set(disabledViews);
  return (
    <div
      role="tablist"
      aria-label="MI workspace view"
      className={cn(
        "flex w-full items-stretch gap-1 rounded-lg border border-navy-600 bg-navy-950/80 p-1 shadow-inner ring-1 ring-inset ring-white/5",
        className,
      )}
    >
      {VIEWS.map(({ id, label, icon: Icon }) => {
        const selected = active === id;
        const isDisabled = disabled.has(id);
        return (
          <button
            key={id}
            type="button"
            role="tab"
            aria-selected={selected}
            aria-disabled={isDisabled}
            disabled={isDisabled}
            title={isDisabled ? "Not applicable for an acquired back book (Funded only)" : undefined}
            onClick={() => !isDisabled && onChange(id)}
            className={cn(
              // Each tab flexes to share the row evenly across the full width.
              "inline-flex flex-1 items-center justify-center gap-1.5 rounded-md px-3 py-1.5 text-[13px] font-medium transition-all",
              isDisabled
                ? "cursor-not-allowed text-ink-600 opacity-40"
                : selected
                  ? "bg-peri-400/20 text-ink-100 shadow-sm ring-1 ring-inset ring-peri-400/50"
                  // Inactive: visibly a clickable chip (subtle fill + border) that
                  // lifts on hover, not near-invisible plain text.
                  : "cursor-pointer bg-navy-800/70 text-ink-300 ring-1 ring-inset ring-white/5 hover:bg-navy-700 hover:text-ink-100 hover:ring-white/10",
            )}
          >
            <Icon size={14} className={selected && !isDisabled ? "text-peri-200" : "text-ink-400"} />
            {label}
          </button>
        );
      })}
    </div>
  );
}
