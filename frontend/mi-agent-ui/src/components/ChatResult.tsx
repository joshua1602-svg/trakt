import { BarChart3, Sheet } from "lucide-react";
import type { Artifact, KPIArtifact, Reconciliation } from "@/domain";
import { isChartArtifact, isTableArtifact, isKPIArtifact } from "@/domain";

/**
 * The result shown INSIDE the conversation. It is deliberately COMPACT — a
 * concise set of key numbers plus links that open the full chart / table in the
 * Artifact Workspace. The chat stays conversational; charts, tables and other
 * artifacts render in the workspace only (never duplicated inline here).
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
  onOpenArtifact,
}: {
  artifacts: Artifact[];
  /** Retained for signature compatibility; pinning happens in the workspace. */
  onTogglePin?: (id: string) => void;
  onOpenArtifact?: (id: string) => void;
}) {
  if (artifacts.length === 0) return null;

  const chart = artifacts.find(isChartArtifact);
  const table = artifacts.find(isTableArtifact);
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
      {(chart || table) && onOpenArtifact && (
        <div className="flex flex-wrap items-center gap-2 text-[11px]">
          {chart && (
            <button
              type="button"
              onClick={() => onOpenArtifact(chart.id)}
              className="inline-flex items-center gap-1.5 rounded-md border border-[var(--color-line)] bg-navy-800/50 px-2.5 py-1 font-medium text-ink-300 hover:border-teal-400/40 hover:text-ink-100"
            >
              <BarChart3 size={12} /> Open chart in workspace
            </button>
          )}
          {table && (
            <button
              type="button"
              onClick={() => onOpenArtifact(table.id)}
              className="inline-flex items-center gap-1.5 rounded-md border border-[var(--color-line)] bg-navy-800/50 px-2.5 py-1 font-medium text-ink-300 hover:border-teal-400/40 hover:text-ink-100"
            >
              <Sheet size={12} /> Open table in workspace
            </button>
          )}
        </div>
      )}
    </div>
  );
}
