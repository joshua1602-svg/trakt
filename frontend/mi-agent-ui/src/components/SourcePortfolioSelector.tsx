import { useEffect, useRef, useState } from "react";
import { Check, ChevronDown, Building2 } from "lucide-react";
import type { SourcePortfolioLens } from "@/domain";
import { cn } from "@/lib/utils";

/**
 * Source-portfolio scope selector — Total / Direct / Acquired / individual
 * cohorts (direct_001, acquired_001, …). Independent of the Funded / Pipeline /
 * Forecast view toggle: this picks *which book*, that picks *which dataset view*.
 * Options are data-driven from `GET /mi/source-portfolios`.
 */
export function SourcePortfolioSelector({
  lenses,
  value,
  onChange,
}: {
  lenses: SourcePortfolioLens[];
  value: string;
  onChange: (lensId: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const active = lenses.find((l) => l.id === value) ?? lenses[0] ?? null;

  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, []);

  // Only Total available → nothing to pick (no provenance in this dataset).
  const single = lenses.length <= 1;

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        aria-label="Source portfolio scope"
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
        <div className="absolute z-30 mt-1.5 w-64 rounded-lg border border-[var(--color-line)] bg-navy-900 p-1 shadow-2xl">
          {lenses.map((l) => (
            <button
              key={l.id}
              type="button"
              onClick={() => {
                onChange(l.id);
                setOpen(false);
              }}
              className="flex w-full items-center gap-2.5 rounded-md px-2.5 py-2 text-left transition-colors hover:bg-navy-800"
            >
              <div className="flex-1">
                <div className="text-[13px] font-medium text-ink-100">{l.label}</div>
                <div className="text-[10px] text-ink-400">
                  {l.kind === "total"
                    ? "Direct + acquired"
                    : l.kind === "type"
                      ? `${l.label} book${l.funded_only ? " · funded only" : ""}`
                      : `${l.source_portfolio_type ?? "cohort"}${l.funded_only ? " · funded only" : ""}`}
                </div>
              </div>
              {l.id === active?.id && <Check size={14} className="text-peri-300" />}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
