import { useState } from "react";
import { BarChart3, Maximize2, Sheet } from "lucide-react";
import type { Artifact, KPIArtifact, Reconciliation } from "@/domain";
import { isChartArtifact, isTableArtifact, isKPIArtifact } from "@/domain";
import { ArtifactCard } from "@/components/ArtifactCard";
import { cn } from "@/lib/utils";

/**
 * The result shown INSIDE the conversation. By default it is COMPACT — a concise
 * narrative is accompanied by key numbers and links that open the full chart/table
 * in the workspace, rather than re-rendering the whole chart in the chat (which
 * duplicated the workspace panel). The operator can expand the full result inline
 * on demand ("Show here"). KPI-only results are already concise, so they render
 * inline directly.
 */
function keyNumbers(artifacts: Artifact[]): { label: string; value: string }[] {
  const kpi = artifacts.find(isKPIArtifact) as KPIArtifact | undefined;
  if (kpi) return kpi.kpis.map((k) => ({ label: k.label, value: k.value }));
  const grouped = artifacts.find(isTableArtifact) ?? artifacts.find(isChartArtifact);
  const out: { label: string; value: string }[] = [];
  const recon = (grouped as { reconciliation?: Reconciliation } | undefined)?.reconciliation;
  const rows = (grouped as { rows?: Array<Record<string, unknown>> } | undefined)?.rows ?? [];
  if (rows.length) out.push({ label: "Groups", value: String(rows.length) });
  if (recon?.coverage_by_balance_pct != null)
    out.push({ label: "Coverage", value: `${recon.coverage_by_balance_pct}%` });
  return out;
}

export function ChatResult({
  artifacts,
  onTogglePin,
  onDrill,
  onAsk,
  onOpenArtifact,
}: {
  artifacts: Artifact[];
  onTogglePin: (id: string) => void;
  onDrill?: (artifact: Artifact, filters: Record<string, unknown>) => void;
  onAsk?: (question: string) => void;
  onOpenArtifact?: (id: string) => void;
}) {
  const chart = artifacts.find(isChartArtifact);
  const table = artifacts.find(isTableArtifact);
  const kpiOnly = !chart && !table && artifacts.some(isKPIArtifact);
  const others = artifacts.filter((a) => a !== chart && a !== table);
  const [view, setView] = useState<"chart" | "table">(chart ? "chart" : "table");
  const [expanded, setExpanded] = useState(false);

  if (artifacts.length === 0) return null;

  const card = (a: Artifact) => (
    <ArtifactCard key={a.id} artifact={a} onTogglePin={onTogglePin} onDrill={onDrill} onAsk={onAsk} />
  );

  // KPI-only results are already concise — render inline.
  if (kpiOnly) {
    return <div className="mt-2 space-y-2">{artifacts.map((a) => card(a))}</div>;
  }

  // Compact (default): key numbers + open-in-workspace links. The full chart/table
  // lives in the workspace; expand inline only on request.
  if (!expanded) {
    const numbers = keyNumbers(artifacts);
    return (
      <div className="mt-2 space-y-2" data-testid="chat-result-compact">
        {numbers.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {numbers.slice(0, 4).map((n) => (
              <div
                key={n.label}
                className="rounded-lg border border-[var(--color-line)] bg-navy-900/50 px-2.5 py-1.5 text-[11px]"
              >
                <span className="text-ink-500">{n.label}: </span>
                <span className="font-semibold text-ink-100">{n.value}</span>
              </div>
            ))}
          </div>
        )}
        <div className="flex flex-wrap items-center gap-2 text-[11px]">
          {chart && onOpenArtifact && (
            <button
              type="button"
              onClick={() => onOpenArtifact(chart.id)}
              className="inline-flex items-center gap-1.5 rounded-md border border-[var(--color-line)] bg-navy-800/50 px-2.5 py-1 font-medium text-ink-300 hover:border-teal-400/40 hover:text-ink-100"
            >
              <BarChart3 size={12} /> Open chart in workspace
            </button>
          )}
          {table && onOpenArtifact && (
            <button
              type="button"
              onClick={() => onOpenArtifact(table.id)}
              className="inline-flex items-center gap-1.5 rounded-md border border-[var(--color-line)] bg-navy-800/50 px-2.5 py-1 font-medium text-ink-300 hover:border-teal-400/40 hover:text-ink-100"
            >
              <Sheet size={12} /> Open table in workspace
            </button>
          )}
          <button
            type="button"
            onClick={() => setExpanded(true)}
            className="inline-flex items-center gap-1.5 rounded-md px-2 py-1 font-medium text-ink-500 hover:text-ink-200"
          >
            <Maximize2 size={12} /> Show here
          </button>
        </div>
      </div>
    );
  }

  // Expanded (explicitly requested): the full result inline, with a Chart/Table
  // toggle when both exist.
  const primary = view === "chart" ? chart ?? table : table ?? chart;
  return (
    <div className="mt-2 space-y-2">
      <div className="flex items-center gap-2">
        {chart && table && (
          <div className="inline-flex items-center gap-0.5 rounded-lg border border-[var(--color-line)] bg-navy-900/60 p-0.5">
            {(
              [
                { id: "chart", label: "Chart", icon: BarChart3 },
                { id: "table", label: "Table", icon: Sheet },
              ] as const
            ).map((m) => (
              <button
                key={m.id}
                type="button"
                onClick={() => setView(m.id)}
                className={cn(
                  "inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-[11px] font-medium transition-colors",
                  view === m.id ? "bg-navy-700 text-ink-100" : "text-ink-400 hover:text-ink-100",
                )}
              >
                <m.icon size={12} />
                {m.label}
              </button>
            ))}
          </div>
        )}
        <button
          type="button"
          onClick={() => setExpanded(false)}
          className="ml-auto rounded-md px-2 py-1 text-[11px] font-medium text-ink-500 hover:text-ink-200"
        >
          Collapse
        </button>
      </div>

      {primary && card(primary)}
      {others.map((a) => card(a))}
    </div>
  );
}
