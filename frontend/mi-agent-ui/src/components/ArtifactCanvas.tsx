import { useMemo, useState } from "react";
import { LayoutList, Rows3, Sparkles } from "lucide-react";
import type { Artifact } from "@/types";
import { ArtifactCard } from "@/components/ArtifactCard";
import { cn } from "@/lib/utils";

type ViewMode = "stack" | "tabs";

export function ArtifactCanvas({
  artifacts,
  onTogglePin,
  isWorking,
}: {
  artifacts: Artifact[];
  onTogglePin: (id: string) => void;
  isWorking: boolean;
}) {
  const [view, setView] = useState<ViewMode>("stack");
  const [activeTab, setActiveTab] = useState(0);

  // Pinned artifacts float to the top of the stack.
  const ordered = useMemo(
    () => [...artifacts].sort((a, b) => Number(!!b.pinned) - Number(!!a.pinned)),
    [artifacts],
  );

  const active = ordered[Math.min(activeTab, ordered.length - 1)];

  return (
    <section className="flex h-full min-w-0 flex-1 flex-col">
      <header className="flex items-center justify-between gap-3 border-b border-[var(--color-line)] px-6 py-3">
        <div>
          <h2 className="text-sm font-semibold text-ink-100">Artifact Workspace</h2>
          <p className="text-xs text-ink-400">
            {artifacts.length} artifact{artifacts.length === 1 ? "" : "s"} · ERM UK — Master
          </p>
        </div>
        <div className="flex items-center gap-1 rounded-lg border border-[var(--color-line)] bg-navy-900/60 p-0.5">
          {(
            [
              { id: "stack", label: "Stack", icon: Rows3 },
              { id: "tabs", label: "Tabs", icon: LayoutList },
            ] as const
          ).map((m) => (
            <button
              key={m.id}
              type="button"
              onClick={() => setView(m.id)}
              className={cn(
                "inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium transition-colors",
                view === m.id
                  ? "bg-navy-700 text-ink-100"
                  : "text-ink-400 hover:text-ink-100",
              )}
            >
              <m.icon size={13} />
              {m.label}
            </button>
          ))}
        </div>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto px-6 py-5">
        {isWorking && (
          <div className="mb-4 flex items-center gap-2 rounded-lg border border-peri-400/20 bg-navy-800/40 px-4 py-3 text-sm text-peri-200">
            <Sparkles size={15} className="text-peri-300" />
            MI Agent is composing artifacts
            <span className="ml-1 inline-flex gap-0.5">
              <span className="dot-1 h-1 w-1 rounded-full bg-peri-300" />
              <span className="dot-2 h-1 w-1 rounded-full bg-peri-300" />
              <span className="dot-3 h-1 w-1 rounded-full bg-peri-300" />
            </span>
          </div>
        )}

        {view === "tabs" && ordered.length > 0 && (
          <div className="mb-4 flex flex-wrap gap-1.5">
            {ordered.map((a, i) => (
              <button
                key={a.id}
                type="button"
                onClick={() => setActiveTab(i)}
                className={cn(
                  "rounded-md border px-2.5 py-1 text-xs font-medium transition-colors",
                  i === Math.min(activeTab, ordered.length - 1)
                    ? "border-peri-400/40 bg-navy-700/60 text-ink-100"
                    : "border-[var(--color-line)] text-ink-400 hover:text-ink-100",
                )}
              >
                {a.title}
              </button>
            ))}
          </div>
        )}

        {view === "stack" ? (
          <div className="flex flex-col gap-4">
            {ordered.map((a) => (
              <div key={a.id} id={`artifact-${a.id}`}>
                <ArtifactCard artifact={a} onTogglePin={onTogglePin} />
              </div>
            ))}
          </div>
        ) : (
          active && (
            <div id={`artifact-${active.id}`}>
              <ArtifactCard artifact={active} onTogglePin={onTogglePin} />
            </div>
          )
        )}

        {ordered.length === 0 && !isWorking && (
          <div className="flex h-64 items-center justify-center text-sm text-ink-500">
            No artifacts yet — ask the MI Agent a question.
          </div>
        )}
      </div>
    </section>
  );
}
