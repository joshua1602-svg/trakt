import { useEffect, useMemo, useState } from "react";
import { ChevronDown, LayoutGrid, LayoutList, Rows3, Trash2 } from "lucide-react";
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
  onClear,
}: {
  artifacts: Artifact[];
  onTogglePin: (id: string) => void;
  isWorking: boolean;
  portfolioName: string;
  onDrill?: (artifact: Artifact, filters: Record<string, unknown>) => void;
  onAsk?: (question: string) => void;
  /** Clear the workspace artifacts (view-only; loaded MI data is untouched). */
  onClear?: () => void;
}) {
  const [view, setView] = useState<ViewMode>("stack");
  const [activeTab, setActiveTab] = useState(0);
  const [filter, setFilter] = useState<ArtifactType | "all">("all");
  // Collapse state persists across sessions (declutter, A8).
  const [collapsed, setCollapsed] = useState<boolean>(
    () => (typeof localStorage !== "undefined"
      && localStorage.getItem("mi.artifactWorkspace.collapsed") === "1"));
  useEffect(() => {
    if (typeof localStorage !== "undefined") {
      localStorage.setItem("mi.artifactWorkspace.collapsed", collapsed ? "1" : "0");
    }
  }, [collapsed]);

  // Pinned float to the top; then apply the type filter.
  const ordered = useMemo(
    () => [...artifacts].sort((a, b) => Number(!!b.pinned) - Number(!!a.pinned)),
    [artifacts],
  );
  const visibleArtifacts = useMemo(
    () => (filter === "all" ? ordered : ordered.filter((a) => a.type === filter)),
    [ordered, filter],
  );
  // Group same-title artifacts into ONE logical artifact so a chart + its table
  // don't render as two duplicate same-name entries. Each group renders as a
  // single card with an internal Chart / Table view toggle.
  const groups = useMemo(() => {
    const map = new Map<string, Artifact[]>();
    for (const a of visibleArtifacts) {
      const key = formatHeading(a.title).toLowerCase();
      const arr = map.get(key);
      if (arr) arr.push(a);
      else map.set(key, [a]);
    }
    return Array.from(map.values());
  }, [visibleArtifacts]);
  const presentTypes = useMemo(
    () => Array.from(new Set(ordered.map((a) => a.type))) as ArtifactType[],
    [ordered],
  );

  const activeGroup = groups[Math.min(activeTab, groups.length - 1)];

  return (
    <section className="flex h-full min-w-0 flex-1 flex-col">
      <header className="flex items-center justify-between gap-3 border-b border-[var(--color-line)] px-6 py-3">
        <div className="flex items-center gap-2.5">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-navy-700 text-peri-300">
            <LayoutGrid size={18} />
          </div>
          <div>
            <h2 className="text-base font-semibold text-ink-100">Artifact Workspace</h2>
            <p className="text-[11px] text-ink-400">
              {artifacts.length} artifact{artifacts.length === 1 ? "" : "s"} · {portfolioName}
            </p>
          </div>
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
          {onClear && artifacts.length > 0 && (
            <button
              type="button"
              onClick={onClear}
              aria-label="Clear artifacts"
              title="Clear artifacts (loaded MI data is untouched)"
              className="inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium text-ink-400 hover:text-rose-300"
            >
              <Trash2 size={13} /> Clear
            </button>
          )}
          <button
            type="button"
            onClick={() => setCollapsed((c) => !c)}
            aria-label={collapsed ? "Expand artifact workspace" : "Collapse artifact workspace"}
            aria-expanded={!collapsed}
            className="inline-flex items-center rounded-md px-1.5 py-1 text-ink-400 hover:text-ink-100"
          >
            <ChevronDown size={15} className={cn("transition-transform", collapsed && "-rotate-90")} />
          </button>
        </div>
      </header>

      {collapsed ? (
        <p className="px-6 py-3 text-[11px] text-ink-500">
          Workspace collapsed — {artifacts.length} artifact{artifacts.length === 1 ? "" : "s"} hidden.
        </p>
      ) : (
      <>
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

        {view === "tabs" && groups.length > 0 && (
          <div className="mb-4 flex flex-wrap gap-1.5">
            {groups.map((g, i) => (
              <button
                key={g[0].id}
                type="button"
                onClick={() => setActiveTab(i)}
                className={cn(
                  "rounded-md border px-2.5 py-1 text-xs font-medium transition-colors",
                  i === Math.min(activeTab, groups.length - 1)
                    ? "border-peri-400/40 bg-navy-700/60 text-ink-100"
                    : "border-[var(--color-line)] text-ink-400 hover:text-ink-100",
                )}
              >
                {formatHeading(g[0].title)}
              </button>
            ))}
          </div>
        )}

        {view === "stack" ? (
          <div className="flex flex-col gap-4">
            {groups.map((g) => (
              <div key={g[0].id}>
                {/* One scroll anchor per member id so "open in workspace" links
                    (which target a specific chart/table id) still land here. */}
                {g.map((a) => (
                  <span key={a.id} id={`artifact-${a.id}`} aria-hidden className="block scroll-mt-4" />
                ))}
                <ArtifactCard artifact={g[0]} views={g} onTogglePin={onTogglePin} onDrill={onDrill} onAsk={onAsk} />
              </div>
            ))}
          </div>
        ) : (
          activeGroup && (
            <div>
              {activeGroup.map((a) => (
                <span key={a.id} id={`artifact-${a.id}`} aria-hidden className="block scroll-mt-4" />
              ))}
              <ArtifactCard artifact={activeGroup[0]} views={activeGroup} onTogglePin={onTogglePin} onDrill={onDrill} onAsk={onAsk} />
            </div>
          )
        )}

        {groups.length === 0 && !isWorking && (
          <EmptyState
            title={filter === "all" ? "No artifacts yet" : `No ${TYPE_LABEL[filter as ArtifactType]} artifacts`}
            hint={filter === "all" ? "Ask the MI Agent a question to generate analysis." : "Try a different filter or ask another question."}
          />
        )}
      </div>
      </>
      )}
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
