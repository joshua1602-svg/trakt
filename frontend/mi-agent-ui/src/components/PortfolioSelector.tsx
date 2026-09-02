import { useEffect, useRef, useState } from "react";
import { Check, ChevronDown, Layers } from "lucide-react";
import type { SnapshotPortfolio } from "@/domain";
import { cn, formatHeading } from "@/lib/utils";

/**
 * What to call this client.
 *
 * A GOVERNED name is rendered exactly as it was approved — "ERE Funding -
 * Equity Release Mortgages" is a name, and title-casing a lowercased copy of it
 * would produce something the client never approved. Only an IDENTIFIER is
 * prettified: "acquired_001" / "CLIENT_001" → "Acquired 001", which is what
 * this surface has always shown for a deployment that declares no name.
 */
function displayName(portfolio?: Pick<SnapshotPortfolio, "label" | "client_name"> | null): string {
  if (!portfolio) return "";
  const governed = portfolio.client_name?.trim();
  if (governed) return governed;
  const label = portfolio.label;
  if (!label) return "";
  return formatHeading(label.toLowerCase()) || label;
}

/**
 * Data-driven CLIENT selector — which onboarded client's platform the workspace
 * is reading (`GET /mi/snapshots`). Only clients discovered from real onboarding
 * output are offered; there are no prototype options.
 *
 * This is NOT a portfolio-scope control. Portfolio scope (Total / Direct /
 * Acquired / an individual source portfolio) has exactly one owner —
 * `PortfolioContextSelector`, driven by the governed portfolio contract. The two
 * are different axes and must never both filter the book.
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
        className="flex max-w-[16rem] items-center gap-2.5 rounded-lg border border-[var(--color-line)] bg-navy-900/60 px-3 py-1.5 text-left transition-colors hover:border-navy-500"
      >
        <Layers size={15} className="text-peri-300" />
        <div className="min-w-0 leading-tight">
          <div className="text-[10px] uppercase tracking-wider text-ink-500">Client</div>
          {/* A governed name can be long; it truncates rather than pushing the
              header around, and carries the full name as a tooltip. */}
          <div
            className="truncate text-[13px] font-medium text-ink-100"
            title={active ? displayName(active) : undefined}
          >
            {active ? displayName(active) : "No client"}
          </div>
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
                <div className="truncate text-[13px] font-medium text-ink-100" title={displayName(p)}>
                  {displayName(p)}
                </div>
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
