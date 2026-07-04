import { useRef, useState } from "react";
import {
  Activity,
  BarChart3,
  Check,
  ChevronDown,
  Copy,
  Download,
  FileWarning,
  FlaskConical,
  LayoutGrid,
  Pin,
  Sheet,
  ShieldCheck,
} from "lucide-react";
import type { Artifact, ArtifactType } from "@/domain";
import { Badge, Card, IconButton } from "@/components/ui";
import { cn, formatHeading, formatTime, toFilenameStem } from "@/lib/utils";
import { ArtifactRenderer } from "@/components/artifacts/ArtifactRenderer";
import { DrillThroughPanel } from "@/components/DrillThroughPanel";
import { ExportMenu } from "@/components/ExportMenu";
import { InsightPanel } from "@/components/InsightPanel";
import { ReconciliationFooter } from "@/components/ReconciliationFooter";
import { isChartArtifact, isTableArtifact } from "@/domain";

const KIND_ICON: Record<ArtifactType, typeof LayoutGrid> = {
  kpi: LayoutGrid,
  chart: BarChart3,
  table: Sheet,
  validation: ShieldCheck,
  risk: Activity,
  scenario: FlaskConical,
  unsupported: FileWarning,
};

const VIEW_LABEL: Record<ArtifactType, string> = {
  kpi: "KPIs",
  chart: "Chart",
  table: "Table",
  validation: "Validation",
  risk: "Risk",
  scenario: "Scenario",
  unsupported: "View",
};

// Preferred order of internal view variants: chart, then table, then the rest.
const VIEW_ORDER: ArtifactType[] = ["chart", "table", "kpi", "risk", "scenario", "validation", "unsupported"];

function orderViews(views: Artifact[]): Artifact[] {
  return [...views].sort((a, b) => VIEW_ORDER.indexOf(a.type) - VIEW_ORDER.indexOf(b.type));
}

export function ArtifactCard({
  artifact,
  views,
  onTogglePin,
  onDrill,
  onAsk,
}: {
  artifact: Artifact;
  /** Sibling view variants of the SAME logical artifact (e.g. the chart + table
   * of "Balance by Age by LTV"). When more than one is present the card shows an
   * internal Chart / Table toggle instead of rendering duplicate cards. */
  views?: Artifact[];
  onTogglePin: (id: string) => void;
  onDrill?: (artifact: Artifact, filters: Record<string, unknown>) => void;
  /** Dispatch a follow-up question (insight investigations), routed via context. */
  onAsk?: (question: string) => void;
}) {
  const [collapsed, setCollapsed] = useState(false);
  const [copied, setCopied] = useState(false);
  const bodyRef = useRef<HTMLDivElement>(null);

  const allViews = orderViews(views && views.length ? views : [artifact]);
  const [viewIdx, setViewIdx] = useState(0);
  const current = allViews[Math.min(viewIdx, allViews.length - 1)] ?? artifact;

  const Icon = KIND_ICON[current.type];
  const exportable = isChartArtifact(current) || isTableArtifact(current);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(JSON.stringify(current, null, 2));
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable in some sandboxes */
    }
  };

  const download = () => {
    const blob = new Blob([JSON.stringify(current, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${toFilenameStem(formatHeading(current.title))}_${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const anyPinned = allViews.some((v) => v.pinned);

  return (
    <Card className={cn("animate-fade-in overflow-hidden", anyPinned && "ring-1 ring-peri-400/30")}>
      <div className="flex items-start gap-3 border-b border-[var(--color-line-soft)] px-4 py-3">
        <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-navy-700/60 text-peri-300">
          <Icon size={15} />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <h3 className="truncate text-sm font-semibold text-ink-100">{formatHeading(current.title)}</h3>
            {anyPinned && <Pin size={12} className="shrink-0 fill-peri-400 text-peri-400" />}
            {current.mock && <Badge tone="amber">Mock</Badge>}
          </div>
          {current.description && (
            <p className="mt-0.5 truncate text-xs text-ink-400">{current.description}</p>
          )}
          <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[10px] font-medium uppercase tracking-wider text-ink-500">
            <span>{current.source.label}</span>
            {current.source.state && (
              <>
                <span className="h-0.5 w-0.5 rounded-full bg-ink-500" />
                <span className="font-mono normal-case">{current.source.state}</span>
              </>
            )}
            <span className="h-0.5 w-0.5 rounded-full bg-ink-500" />
            <span>{formatTime(current.createdAt)}</span>
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-1.5">
          {/* Internal view toggle — one artifact, Chart / Table (etc.) variants. */}
          {allViews.length > 1 && (
            <div
              role="tablist"
              aria-label="Artifact view"
              className="inline-flex items-center gap-0.5 rounded-lg border border-[var(--color-line)] bg-navy-950/60 p-0.5"
            >
              {allViews.map((v, i) => {
                const VIcon = KIND_ICON[v.type];
                const selected = i === Math.min(viewIdx, allViews.length - 1);
                return (
                  <button
                    key={v.id}
                    type="button"
                    role="tab"
                    aria-selected={selected}
                    onClick={() => setViewIdx(i)}
                    className={cn(
                      "inline-flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium transition-colors",
                      selected
                        ? "bg-peri-400/20 text-ink-100 ring-1 ring-inset ring-peri-400/40"
                        : "text-ink-400 hover:text-ink-100",
                    )}
                  >
                    <VIcon size={12} />
                    {VIEW_LABEL[v.type]}
                  </button>
                );
              })}
            </div>
          )}
          <div className="flex items-center gap-0.5">
            <IconButton label={current.pinned ? "Unpin" : "Pin"} active={current.pinned} onClick={() => onTogglePin(current.id)}>
              <Pin size={14} />
            </IconButton>
            <IconButton label={copied ? "Copied" : "Copy data"} onClick={copy}>
              {copied ? <Check size={14} className="text-mint-400" /> : <Copy size={14} />}
            </IconButton>
            {exportable ? (
              <ExportMenu artifact={current} bodyRef={bodyRef} onJson={download} />
            ) : (
              <IconButton label="Download JSON" onClick={download}>
                <Download size={14} />
              </IconButton>
            )}
            <IconButton label={collapsed ? "Expand" : "Collapse"} onClick={() => setCollapsed((c) => !c)}>
              <ChevronDown size={14} className={cn("transition-transform", collapsed && "-rotate-90")} />
            </IconButton>
          </div>
        </div>
      </div>

      {!collapsed && (
        <div className="p-4" ref={bodyRef}>
          <ArtifactRenderer artifact={current} />
          {(isChartArtifact(current) || isTableArtifact(current)) && (
            <>
              <InsightPanel artifact={current} onAsk={onAsk} />
              <DrillThroughPanel
                artifact={current}
                onDrill={onDrill ? (filters) => onDrill(current, filters) : undefined}
              />
            </>
          )}
          <ReconciliationFooter artifact={current} />
          {current.warnings && current.warnings.length > 0 && (
            <div className="mt-3 rounded-lg border border-amber-400/20 bg-amber-400/5 px-3 py-2 text-[11px] text-amber-300/90">
              {current.warnings.map((w, i) => (
                <div key={i}>⚠ {w}</div>
              ))}
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
