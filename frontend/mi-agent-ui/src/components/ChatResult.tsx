import type { ReactNode } from "react";
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
  workspaceArtifactIds,
}: {
  artifacts: Artifact[];
  /** Retained for signature compatibility; pinning happens in the workspace. */
  onTogglePin?: (id: string) => void;
  onOpenArtifact?: (id: string) => void;
  /** Ids still present in the workspace; links to cleared artifacts go stale. */
  workspaceArtifactIds?: Set<string>;
}) {
  if (artifacts.length === 0) return null;

  const chart = artifacts.find(isChartArtifact);
  const table = artifacts.find(isTableArtifact);
  const numbers = keyNumbers(artifacts);
  const isStale = (id: string) =>
    workspaceArtifactIds ? !workspaceArtifactIds.has(id) : false;

  const openButton = (id: string, icon: ReactNode, label: string) => {
    const stale = isStale(id);
    return (
      <button
        type="button"
        disabled={stale}
        title={stale
          ? "No longer in the workspace — ask the question again to regenerate it"
          : undefined}
        onClick={() => !stale && onOpenArtifact?.(id)}
        className={
          stale
            ? "inline-flex cursor-not-allowed items-center gap-1.5 rounded-md border border-[var(--color-line)] bg-navy-800/50 px-2.5 py-1 font-medium text-ink-300 opacity-50"
            : "inline-flex items-center gap-1.5 rounded-md border border-[var(--color-line)] bg-navy-800/50 px-2.5 py-1 font-medium text-ink-300 hover:border-teal-400/40 hover:text-ink-100"
        }
      >
        {icon} {stale ? `${label} (cleared)` : label}
      </button>
    );
  };

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
          {chart && openButton(chart.id, <BarChart3 size={12} />, "Open chart in workspace")}
          {table && openButton(table.id, <Sheet size={12} />, "Open table in workspace")}
        </div>
      )}
    </div>
  );
}
