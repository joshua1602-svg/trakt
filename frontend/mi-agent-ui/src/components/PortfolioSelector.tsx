import { useEffect, useRef, useState } from "react";
import { Check, ChevronDown, Layers } from "lucide-react";
import { PORTFOLIOS } from "@/data/catalog";
import { cn } from "@/lib/utils";

export function PortfolioSelector({
  value,
  onChange,
}: {
  value: string;
  onChange: (id: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const active = PORTFOLIOS.find((p) => p.id === value) ?? PORTFOLIOS[0];

  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, []);

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-2.5 rounded-lg border border-[var(--color-line)] bg-navy-900/60 px-3 py-1.5 text-left transition-colors hover:border-navy-500"
      >
        <Layers size={15} className="text-peri-300" />
        <div className="leading-tight">
          <div className="text-[13px] font-medium text-ink-100">{active.name}</div>
          <div className="text-[10px] text-ink-400">{active.entity}</div>
        </div>
        <ChevronDown
          size={14}
          className={cn("text-ink-400 transition-transform", open && "rotate-180")}
        />
      </button>

      {open && (
        <div className="absolute z-30 mt-1.5 w-64 rounded-lg border border-[var(--color-line)] bg-navy-900 p-1 shadow-2xl">
          {PORTFOLIOS.map((p) => (
            <button
              key={p.id}
              type="button"
              onClick={() => {
                onChange(p.id);
                setOpen(false);
              }}
              className="flex w-full items-center gap-2.5 rounded-md px-2.5 py-2 text-left transition-colors hover:bg-navy-800"
            >
              <div className="flex-1">
                <div className="text-[13px] font-medium text-ink-100">{p.name}</div>
                <div className="text-[10px] text-ink-400">{p.entity}</div>
              </div>
              {p.id === value && <Check size={14} className="text-peri-300" />}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
