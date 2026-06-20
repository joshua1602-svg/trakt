import { ArrowDownRight, ArrowUpRight, Minus } from "lucide-react";
import type { KPIArtifactData } from "@/types";
import { cn } from "@/lib/utils";

export function KPIGrid({ data }: { data: KPIArtifactData }) {
  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-3">
      {data.kpis.map((kpi) => {
        const Icon =
          kpi.trend === "up"
            ? ArrowUpRight
            : kpi.trend === "down"
              ? ArrowDownRight
              : Minus;
        const deltaColor =
          kpi.deltaIntent === "positive"
            ? "text-mint-400"
            : kpi.deltaIntent === "negative"
              ? "text-rose-400"
              : "text-ink-400";
        return (
          <div
            key={kpi.id}
            className="rounded-lg border border-[var(--color-line-soft)] bg-navy-850/60 p-3.5"
          >
            <div className="text-[11px] font-medium uppercase tracking-wider text-ink-400">
              {kpi.label}
            </div>
            <div className="mt-1.5 font-mono text-2xl font-semibold tabular-nums text-ink-100">
              {kpi.value}
            </div>
            <div className="mt-1.5 flex items-center gap-1.5">
              {kpi.delta && (
                <span
                  className={cn(
                    "inline-flex items-center gap-0.5 text-xs font-medium",
                    deltaColor,
                  )}
                >
                  <Icon size={13} strokeWidth={2.5} />
                  {kpi.delta}
                </span>
              )}
              {kpi.hint && (
                <span className="text-[11px] text-ink-500">{kpi.hint}</span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
