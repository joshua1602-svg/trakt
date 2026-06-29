import { Activity, GitBranch, Landmark, ShieldAlert, TrendingUp } from "lucide-react";
import type { WorkspaceView } from "@/domain";
import { cn } from "@/lib/utils";

const VIEWS: { id: WorkspaceView; label: string; icon: typeof Landmark }[] = [
  { id: "funded", label: "Funded", icon: Landmark },
  { id: "pipeline", label: "Pipeline", icon: GitBranch },
  { id: "forecast", label: "Forecast", icon: TrendingUp },
  { id: "evolution", label: "Evolution", icon: Activity },
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
}: {
  active: WorkspaceView;
  onChange: (view: WorkspaceView) => void;
  disabledViews?: WorkspaceView[];
}) {
  const disabled = new Set(disabledViews);
  return (
    <div
      role="tablist"
      aria-label="MI workspace view"
      className="inline-flex items-center gap-1 rounded-lg border border-[var(--color-line)] bg-navy-900/60 p-1"
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
              "inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-[13px] font-medium transition-colors",
              isDisabled
                ? "cursor-not-allowed text-ink-600 opacity-40"
                : selected
                  ? "bg-navy-700/80 text-ink-100 shadow-sm"
                  : "text-ink-400 hover:text-ink-200",
            )}
          >
            <Icon size={14} className={selected && !isDisabled ? "text-peri-300" : "text-ink-500"} />
            {label}
          </button>
        );
      })}
    </div>
  );
}
