import { useEffect, useRef, useState } from "react";
import { Building2, Check, ChevronDown } from "lucide-react";
import type { PortfolioContextOption } from "@/domain";
import { cn } from "@/lib/utils";

/**
 * THE portfolio selector — the single control that sets workspace scope.
 *
 * It renders the governed hierarchy exactly as the backend resolved it:
 *
 *     Total
 *     ├── Direct
 *     │   ├── direct_001
 *     │   └── direct_002
 *     └── Acquired
 *         ├── acquired_001
 *         └── acquired_002
 *
 * Every option, its depth and its membership come from `GET /mi/portfolio-context`.
 * Onboarding a new portfolio changes this list with no change here: there is no
 * hard-coded id, no prefix parsing and no client-side grouping in this file.
 *
 * The secondary text under each option is likewise backend-decided — the count
 * of member portfolios and the capability explanation the resolver wrote.
 */
export function PortfolioContextSelector({
  contexts,
  value,
  onChange,
}: {
  contexts: PortfolioContextOption[];
  value: string;
  onChange: (contextId: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const active = contexts.find((c) => c.context_id === value) ?? contexts[0] ?? null;

  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, []);

  // One option (or none) → nothing to choose: the platform has a single book.
  const single = contexts.length <= 1;

  const subtitle = (c: PortfolioContextOption): string => {
    if (c.context_kind === "portfolio") {
      return c.portfolio_types[0] ?? "source portfolio";
    }
    const n = c.portfolio_ids.length;
    const label = c.context_kind === "total" ? "portfolio" : `${c.label.toLowerCase()} portfolio`;
    return `${n} ${label}${n === 1 ? "" : "s"}`;
  };

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        aria-label="Portfolio scope"
        aria-haspopup="listbox"
        aria-expanded={open}
        data-testid="portfolio-context-selector"
        onClick={() => !single && setOpen((o) => !o)}
        className="flex items-center gap-2.5 rounded-lg border border-[var(--color-line)] bg-navy-900/60 px-3 py-1.5 text-left transition-colors hover:border-navy-500"
      >
        <Building2 size={15} className="text-peri-300" />
        <div className="leading-tight">
          <div className="text-[10px] uppercase tracking-wider text-ink-500">Portfolio</div>
          <div className="text-[13px] font-medium text-ink-100">{active?.label ?? "Total"}</div>
        </div>
        {!single && (
          <ChevronDown size={14} className={cn("text-ink-400 transition-transform", open && "rotate-180")} />
        )}
      </button>

      {open && !single && (
        <div
          role="listbox"
          aria-label="Portfolio scope options"
          className="absolute z-30 mt-1.5 w-72 rounded-lg border border-[var(--color-line)] bg-navy-900 p-1 shadow-2xl"
        >
          {contexts.map((c) => (
            <button
              key={c.context_id}
              type="button"
              role="option"
              aria-selected={c.context_id === active?.context_id}
              data-context-id={c.context_id}
              data-depth={c.depth}
              onClick={() => {
                onChange(c.context_id);
                setOpen(false);
              }}
              className="flex w-full items-center gap-2.5 rounded-md py-2 pr-2.5 text-left transition-colors hover:bg-navy-800"
              // Indentation renders the governed depth; it never computes it.
              style={{ paddingLeft: `${10 + c.depth * 14}px` }}
            >
              <div className="min-w-0 flex-1">
                <div className={cn("truncate text-[13px] text-ink-100",
                  c.depth === 0 ? "font-semibold" : c.depth === 1 ? "font-medium" : "font-normal")}>
                  {c.label}
                </div>
                <div className="truncate text-[10px] text-ink-400">{subtitle(c)}</div>
              </div>
              {c.context_id === active?.context_id && <Check size={14} className="text-peri-300" />}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
