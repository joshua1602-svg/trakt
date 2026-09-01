import { useState } from "react";
import { GitBranch, Info, Landmark, ShieldAlert, TrendingUp, X } from "lucide-react";
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

const VIEW_LABEL: Record<string, string> = Object.fromEntries(
  VIEWS.map((v) => [v.id, v.label]),
);

/** Segmented control selecting the active MI workspace view.
 *
 * ``disabledViews`` and ``disabledReasons`` come from the GOVERNED CAPABILITY
 * CONTRACT for the selected portfolio context — this component does not decide
 * that an acquired book has no pipeline, or that any scope lacks a forecast. It
 * renders the backend's decision and shows the backend's explanation. The
 * reason is available on hover (title) AND inline: clicking/tapping a disabled
 * tab reveals the business reason below the tab row, so keyboard and touch
 * users are not left with a dead control and no explanation.
 */
export function ViewToggle({
  active,
  onChange,
  disabledViews = [],
  disabledReasons = {},
  onPrefetch,
  className,
}: {
  active: WorkspaceView;
  onChange: (view: WorkspaceView) => void;
  disabledViews?: WorkspaceView[];
  /** Backend-authored business reason per disabled view, shown on hover. */
  disabledReasons?: Record<string, string>;
  /** Warm a view's data on hover/focus, BEFORE the click. Optional and
   *  best-effort: the caller de-duplicates, so pointing at a tab several times
   *  issues at most one request, and never issues one for a disabled view. */
  onPrefetch?: (view: WorkspaceView) => void;
  /** Optional wrapper classes (e.g. `flex-1` to span the dashboard width). */
  className?: string;
}) {
  const disabled = new Set(disabledViews);
  // Which disabled view's reason is revealed inline (dismissible).
  const [revealedReason, setRevealedReason] = useState<WorkspaceView | null>(null);
  const revealed = revealedReason && disabled.has(revealedReason) ? revealedReason : null;

  return (
    <div className={cn("min-w-0", className)}>
      <div
        role="tablist"
        aria-label="MI workspace view"
        className="nav-primary w-full"
      >
        {VIEWS.map(({ id, label, icon: Icon }) => {
          const selected = active === id;
          const isDisabled = disabled.has(id);
          return (
            // The wrapper catches clicks a disabled button swallows, so tapping
            // a dead tab still reveals WHY it is unavailable (never onChange).
            <span
              key={id}
              className="min-w-0 flex-1"
              onClick={isDisabled ? () => setRevealedReason((r) => (r === id ? null : id)) : undefined}
            >
              <button
                type="button"
                role="tab"
                aria-selected={selected}
                aria-disabled={isDisabled}
                disabled={isDisabled}
                title={isDisabled
                  ? (disabledReasons[id]
                     ?? "Not applicable for the selected portfolio scope.")
                  : undefined}
                data-disabled-reason={isDisabled ? (disabledReasons[id] ?? undefined) : undefined}
                onClick={() => !isDisabled && onChange(id)}
                onMouseEnter={!isDisabled && onPrefetch ? () => onPrefetch(id) : undefined}
                onFocus={!isDisabled && onPrefetch ? () => onPrefetch(id) : undefined}
                className={cn(
                  // Each tab flexes to share the row evenly across the full width.
                  // `nav-primary-item` carries the whole tier-1 treatment: the
                  // accent underline that marks selection, the hover lift and
                  // the disabled ink — see index.css. Selection is read off
                  // aria-selected, so the visual state cannot drift from the
                  // accessible one.
                  "nav-primary-item w-full",
                  isDisabled ? "cursor-not-allowed" : "cursor-pointer",
                )}
              >
                <Icon size={15} strokeWidth={selected && !isDisabled ? 2.25 : 1.75}
                  className={selected && !isDisabled ? "text-peri-200" : "text-current opacity-70"} />
                {label}
              </button>
            </span>
          );
        })}
      </div>

      {revealed && (
        <div
          role="note"
          data-testid="disabled-view-reason"
          className="mt-[var(--gap-tight)] flex items-start gap-2 rounded-md border-l-2 border-l-amber-400/50 bg-navy-850/70 px-3 py-2 text-[var(--fs-label)] leading-relaxed text-ink-300"
        >
          <Info size={13} className="mt-0.5 shrink-0 text-amber-400" />
          <span className="min-w-0 flex-1">
            <span className="font-semibold text-ink-100">{VIEW_LABEL[revealed]} unavailable · </span>
            {disabledReasons[revealed] ?? "Not applicable for the selected portfolio scope."}
          </span>
          <button
            type="button"
            aria-label="Dismiss"
            onClick={() => setRevealedReason(null)}
            className="shrink-0 rounded p-0.5 text-ink-500 transition-colors hover:text-ink-100"
          >
            <X size={12} />
          </button>
        </div>
      )}
    </div>
  );
}
