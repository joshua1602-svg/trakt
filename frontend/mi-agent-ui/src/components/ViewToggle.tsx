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

/** Segmented control selecting the active MI workspace view. */
export function ViewToggle({
  active,
  onChange,
}: {
  active: WorkspaceView;
  onChange: (view: WorkspaceView) => void;
}) {
  return (
    <div
      role="tablist"
      aria-label="MI workspace view"
      className="inline-flex items-center gap-1 rounded-lg border border-[var(--color-line)] bg-navy-900/60 p-1"
    >
      {VIEWS.map(({ id, label, icon: Icon }) => {
        const selected = active === id;
        return (
          <button
            key={id}
            type="button"
            role="tab"
            aria-selected={selected}
            onClick={() => onChange(id)}
            className={cn(
              "inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-[13px] font-medium transition-colors",
              selected
                ? "bg-navy-700/80 text-ink-100 shadow-sm"
                : "text-ink-400 hover:text-ink-200",
            )}
          >
            <Icon size={14} className={selected ? "text-peri-300" : "text-ink-500"} />
            {label}
          </button>
        );
      })}
    </div>
  );
}
