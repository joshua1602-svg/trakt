import type { MigrationCell, RagStatus, RiskArtifact, RiskGroup } from "@/domain";
import { THEME } from "@/lib/theme";
import { cn, formatGBP } from "@/lib/utils";

const RAG_TEXT: Record<RagStatus, string> = {
  green: "text-mint-400",
  amber: "text-amber-400",
  red: "text-rose-400",
  below_minimum: "text-ink-500",
};
const RAG_BAR: Record<RagStatus, string> = {
  green: THEME.rag.green,
  amber: THEME.rag.amber,
  red: THEME.rag.red,
  below_minimum: THEME.rag.below_minimum,
};
const RAG_LABEL: Record<RagStatus, string> = {
  green: "Within limit",
  amber: "Approaching",
  red: "Breach",
  below_minimum: "Below min",
};

function LimitRow({ g }: { g: RiskGroup }) {
  const sharePct = g.share * 100;
  const limitPct = g.limit ? g.limit * 100 : undefined;
  const usage = g.limit ? Math.min((g.share / g.limit) * 100, 100) : sharePct;
  return (
    <div className="rounded-lg border border-[var(--color-line-soft)] bg-navy-850/40 p-3">
      <div className="flex items-center justify-between gap-2">
        <span className="truncate text-sm font-medium text-ink-100">{g.name}</span>
        <span className={cn("shrink-0 text-[10px] font-semibold uppercase tracking-wider", RAG_TEXT[g.status])}>
          {RAG_LABEL[g.status]}
        </span>
      </div>
      <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-navy-800">
        <span className="block h-full rounded-full" style={{ width: `${usage}%`, background: RAG_BAR[g.status] }} />
      </div>
      <div className="mt-1.5 flex items-center justify-between font-mono text-[11px] text-ink-400">
        <span>
          {formatGBP(g.balance * 1e6)} · {sharePct.toFixed(1)}%
        </span>
        {limitPct != null && <span className="text-ink-500">limit {limitPct.toFixed(0)}%</span>}
      </div>
    </div>
  );
}

function MigrationMatrix({ artifact }: { artifact: RiskArtifact }) {
  const axis = artifact.axis ?? [];
  const lookup = new Map<string, MigrationCell>();
  for (const c of artifact.matrix ?? []) lookup.set(`${c.from}->${c.to}`, c);
  const max = Math.max(...(artifact.matrix ?? []).map((c) => c.share), 0.0001);

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-xs">
        <thead>
          <tr>
            <th className="p-1.5 text-left text-[10px] font-semibold uppercase tracking-wider text-ink-500">From \ To</th>
            {axis.map((a) => (
              <th key={a} className="p-1.5 text-center font-mono font-medium text-ink-400">{a}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {axis.map((from) => (
            <tr key={from}>
              <td className="p-1.5 font-mono font-medium text-ink-400">{from}</td>
              {axis.map((to) => {
                const c = lookup.get(`${from}->${to}`);
                const intensity = c ? c.share / max : 0;
                const isDiag = from === to;
                const color = c
                  ? c.movement === "deteriorated"
                    ? THEME.negative
                    : c.movement === "improved"
                      ? THEME.positive
                      : THEME.peri
                  : "transparent";
                return (
                  <td key={to} className="p-0.5">
                    <div
                      className={cn(
                        "flex h-9 items-center justify-center rounded font-mono text-[10px]",
                        c ? "text-white" : "text-ink-600",
                        isDiag && "ring-1 ring-inset ring-white/10",
                      )}
                      style={{ background: c ? `${color}${Math.round(40 + intensity * 200).toString(16).padStart(2, "0")}` : "rgba(255,255,255,0.02)" }}
                      title={c ? `${from}→${to}: ${(c.share * 100).toFixed(1)}% (${c.movement})` : undefined}
                    >
                      {c ? `${(c.share * 100).toFixed(1)}` : "·"}
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[10px] text-ink-400">
        <span className="inline-flex items-center gap-1.5"><span className="h-2 w-2 rounded-full" style={{ background: THEME.positive }} />Improved</span>
        <span className="inline-flex items-center gap-1.5"><span className="h-2 w-2 rounded-full" style={{ background: THEME.peri }} />Unchanged</span>
        <span className="inline-flex items-center gap-1.5"><span className="h-2 w-2 rounded-full" style={{ background: THEME.negative }} />Deteriorated</span>
        <span className="ml-auto text-ink-500">% of balance</span>
      </div>
    </div>
  );
}

export function RiskArtifactView({ artifact }: { artifact: RiskArtifact }) {
  if (artifact.mode === "migration") return <MigrationMatrix artifact={artifact} />;
  return (
    <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
      {(artifact.groups ?? []).map((g) => (
        <LimitRow key={g.name} g={g} />
      ))}
    </div>
  );
}
