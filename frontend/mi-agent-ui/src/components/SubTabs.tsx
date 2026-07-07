import { cn } from "@/lib/utils";

/** A lightweight sub-tab bar used inside a top-level dashboard workspace
 * (Funded / Pipeline / Forecast) to switch between its sub-views. Visually
 * subordinate to the top-level ViewToggle. */
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
      className="inline-flex flex-wrap items-center gap-1 rounded-lg border border-navy-600 bg-navy-950/80 p-1 ring-1 ring-inset ring-white/5"
    >
      {tabs.map((t) => (
        <button
          key={t.id}
          type="button"
          role="tab"
          aria-selected={active === t.id}
          onClick={() => onChange(t.id)}
          className={cn(
            "rounded-md px-3 py-1 text-[12px] font-medium transition-all",
            active === t.id
              ? "bg-peri-400/20 text-ink-100 ring-1 ring-inset ring-peri-400/50"
              : "cursor-pointer bg-navy-800/70 text-ink-300 ring-1 ring-inset ring-white/5 hover:bg-navy-700 hover:text-ink-100",
          )}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}
