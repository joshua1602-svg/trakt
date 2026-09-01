import { cn } from "@/lib/utils";

/** A lightweight sub-tab bar used inside a top-level dashboard workspace
 * (Funded / Pipeline / Forecast) to switch between its sub-views. Tier 2 of
 * three: visually subordinate to the ViewToggle rail above it and dominant
 * over any unit switch inside a panel. */
export function SubTabs<T extends string>({
  tabs, active, onChange, ariaLabel, testId,
}: {
  tabs: { id: T; label: string }[];
  active: T;
  onChange: (id: T) => void;
  ariaLabel: string;
  testId?: string;
}) {
  return (
    <div
      role="tablist"
      aria-label={ariaLabel}
      data-testid={testId}
      className="nav-secondary"
    >
      {tabs.map((t) => (
        <button
          key={t.id}
          type="button"
          role="tab"
          aria-selected={active === t.id}
          onClick={() => onChange(t.id)}
          // `nav-secondary-item` carries the tier-2 treatment: a recessed
          // well with the selected view raised out of it as a solid thumb.
          // A contained control, deliberately a different MECHANISM from the
          // tier-1 rail above it, so rank is legible without reading labels.
          className={cn("nav-secondary-item", active !== t.id && "cursor-pointer")}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}
