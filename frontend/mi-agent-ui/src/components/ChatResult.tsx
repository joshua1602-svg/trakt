import { useState } from "react";
import { BarChart3, Sheet } from "lucide-react";
import type { Artifact } from "@/domain";
import { isChartArtifact, isTableArtifact } from "@/domain";
import { ArtifactCard } from "@/components/ArtifactCard";
import { cn } from "@/lib/utils";

/**
 * The result rendered directly INSIDE the conversation: the chart/table (with a
 * Chart/Table toggle when both exist), its key observations, drill-through and
 * downloads — by reusing ArtifactCard. This makes the answer visible in the chat
 * thread itself, not just as a link to a separate canvas.
 */
export function ChatResult({
  artifacts,
  onTogglePin,
  onDrill,
  onAsk,
}: {
  artifacts: Artifact[];
  onTogglePin: (id: string) => void;
  onDrill?: (artifact: Artifact, filters: Record<string, unknown>) => void;
  onAsk?: (question: string) => void;
}) {
  const chart = artifacts.find(isChartArtifact);
  const table = artifacts.find(isTableArtifact);
  const others = artifacts.filter((a) => a !== chart && a !== table);
  const [view, setView] = useState<"chart" | "table">(chart ? "chart" : "table");

  if (artifacts.length === 0) return null;

  const primary = view === "chart" ? chart ?? table : table ?? chart;
  const card = (a: Artifact) => (
    <ArtifactCard key={a.id} artifact={a} onTogglePin={onTogglePin} onDrill={onDrill} onAsk={onAsk} />
  );

  return (
    <div className="mt-2 space-y-2">
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

      {primary && card(primary)}
      {others.map((a) => card(a))}
    </div>
  );
}
