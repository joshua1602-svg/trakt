import { useEffect, useRef, useState } from "react";
import { Check, ChevronDown, Layers } from "lucide-react";
import type { SnapshotPortfolio } from "@/domain";
import { cn } from "@/lib/utils";

/**
 * Data-driven funded-portfolio selector. Only portfolios discovered from real
 * onboarding output (`GET /mi/snapshots`) are offered — no prototype options.
 */
export function PortfolioSelector({
  portfolios,
  value,
  onChange,
}: {
  portfolios: SnapshotPortfolio[];
  value: string | null;
  onChange: (clientId: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const active = portfolios.find((p) => p.client_id === value) ?? portfolios[0] ?? null;

  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, []);

  const single = portfolios.length <= 1;

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => !single && setOpen((o) => !o)}
        className="flex items-center gap-2.5 rounded-lg border border-[var(--color-line)] bg-navy-900/60 px-3 py-1.5 text-left transition-colors hover:border-navy-500"
      >
        <Layers size={15} className="text-peri-300" />
        <div className="leading-tight">
          <div className="text-[10px] uppercase tracking-wider text-ink-500">Funded Portfolio</div>
          <div className="text-[13px] font-medium text-ink-100">{active?.label ?? "No portfolio"}</div>
        </div>
        {!single && (
          <ChevronDown size={14} className={cn("text-ink-400 transition-transform", open && "rotate-180")} />
        )}
      </button>

      {open && !single && (
        <div className="absolute z-30 mt-1.5 w-64 rounded-lg border border-[var(--color-line)] bg-navy-900 p-1 shadow-2xl">
          {portfolios.map((p) => (
            <button
              key={p.client_id}
              type="button"
              onClick={() => {
                onChange(p.client_id);
                setOpen(false);
              }}
              className="flex w-full items-center gap-2.5 rounded-md px-2.5 py-2 text-left transition-colors hover:bg-navy-800"
            >
              <div className="flex-1">
                <div className="text-[13px] font-medium text-ink-100">{p.label}</div>
                <div className="text-[10px] text-ink-400">{p.runs.length} reporting run{p.runs.length === 1 ? "" : "s"}</div>
              </div>
              {p.client_id === active?.client_id && <Check size={14} className="text-peri-300" />}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
