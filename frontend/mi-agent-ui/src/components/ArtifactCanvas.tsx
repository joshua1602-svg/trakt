import { useMemo, useState } from "react";
import { LayoutList, Rows3 } from "lucide-react";
import type { Artifact, ArtifactType } from "@/domain";
import { ArtifactCard } from "@/components/ArtifactCard";
import { EmptyState, LoadingState } from "@/components/states/States";
import { cn, formatHeading } from "@/lib/utils";

type ViewMode = "stack" | "tabs";

const TYPE_LABEL: Record<ArtifactType, string> = {
  kpi: "KPIs",
  chart: "Charts",
  table: "Tables",
  validation: "Validation",
  risk: "Risk",
  scenario: "Scenario",
  unsupported: "Unsupported",
};

export function ArtifactCanvas({
  artifacts,
  onTogglePin,
  isWorking,
  portfolioName,
  onDrill,
  onAsk,
}: {
  artifacts: Artifact[];
  onTogglePin: (id: string) => void;
  isWorking: boolean;
  portfolioName: string;
  onDrill?: (artifact: Artifact, filters: Record<string, unknown>) => void;
  onAsk?: (question: string) => void;
}) {
  const [view, setView] = useState<ViewMode>("stack");
  const [activeTab, setActiveTab] = useState(0);
  const [filter, setFilter] = useState<ArtifactType | "all">("all");

  // Pinned float to the top; then apply the type filter.
  const ordered = useMemo(
    () => [...artifacts].sort((a, b) => Number(!!b.pinned) - Number(!!a.pinned)),
    [artifacts],
  );
  const visible = useMemo(
    () => (filter === "all" ? ordered : ordered.filter((a) => a.type === filter)),
    [ordered, filter],
  );
  const presentTypes = useMemo(
    () => Array.from(new Set(ordered.map((a) => a.type))) as ArtifactType[],
    [ordered],
  );

  const active = visible[Math.min(activeTab, visible.length - 1)];

  return (
    <section className="flex h-full min-w-0 flex-1 flex-col">
      <header className="flex items-center justify-between gap-3 border-b border-[var(--color-line)] px-6 py-3">
        <div>
          <h2 className="text-sm font-semibold text-ink-100">Artifact Workspace</h2>
          <p className="text-xs text-ink-400">
            {artifacts.length} artifact{artifacts.length === 1 ? "" : "s"} · {portfolioName}
          </p>
        </div>
        <div className="flex items-center gap-1 rounded-lg border border-[var(--color-line)] bg-navy-900/60 p-0.5">
          {([
            { id: "stack", label: "Stack", icon: Rows3 },
            { id: "tabs", label: "Tabs", icon: LayoutList },
          ] as const).map((m) => (
            <button
              key={m.id}
              type="button"
              onClick={() => setView(m.id)}
              className={cn(
                "inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium transition-colors",
                view === m.id ? "bg-navy-700 text-ink-100" : "text-ink-400 hover:text-ink-100",
              )}
            >
              <m.icon size={13} />
              {m.label}
            </button>
          ))}
        </div>
      </header>

      {presentTypes.length > 1 && (
        <div className="flex flex-wrap items-center gap-1.5 border-b border-[var(--color-line-soft)] px-6 py-2">
          <FilterChip label="All" active={filter === "all"} onClick={() => setFilter("all")} />
          {presentTypes.map((t) => (
            <FilterChip key={t} label={TYPE_LABEL[t]} active={filter === t} onClick={() => setFilter(t)} />
          ))}
        </div>
      )}

      <div className="min-h-0 flex-1 overflow-y-auto px-6 py-5">
        {isWorking && <LoadingState />}

        {view === "tabs" && visible.length > 0 && (
          <div className="mb-4 flex flex-wrap gap-1.5">
            {visible.map((a, i) => (
              <button
                key={a.id}
                type="button"
                onClick={() => setActiveTab(i)}
                className={cn(
                  "rounded-md border px-2.5 py-1 text-xs font-medium transition-colors",
                  i === Math.min(activeTab, visible.length - 1)
                    ? "border-peri-400/40 bg-navy-700/60 text-ink-100"
                    : "border-[var(--color-line)] text-ink-400 hover:text-ink-100",
                )}
              >
                {formatHeading(a.title)}
              </button>
            ))}
          </div>
        )}

        {view === "stack" ? (
          <div className="flex flex-col gap-4">
            {visible.map((a) => (
              <div key={a.id} id={`artifact-${a.id}`}>
                <ArtifactCard artifact={a} onTogglePin={onTogglePin} onDrill={onDrill} onAsk={onAsk} />
              </div>
            ))}
          </div>
        ) : (
          active && (
            <div id={`artifact-${active.id}`}>
              <ArtifactCard artifact={active} onTogglePin={onTogglePin} onDrill={onDrill} onAsk={onAsk} />
            </div>
          )
        )}

        {visible.length === 0 && !isWorking && (
          <EmptyState
            title={filter === "all" ? "No artifacts yet" : `No ${TYPE_LABEL[filter as ArtifactType]} artifacts`}
            hint={filter === "all" ? "Ask the MI Agent a question to generate analysis." : "Try a different filter or ask another question."}
          />
        )}
      </div>
    </section>
  );
}

function FilterChip({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "rounded-full border px-2.5 py-0.5 text-[11px] font-medium transition-colors",
        active
          ? "border-peri-400/40 bg-navy-700/60 text-peri-200"
          : "border-[var(--color-line)] text-ink-400 hover:text-ink-100",
      )}
    >
      {label}
    </button>
  );
}
