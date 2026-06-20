import { useState } from "react";
import {
  BarChart3,
  Check,
  ChevronDown,
  Copy,
  Download,
  Pin,
  Sheet,
  ShieldCheck,
  LayoutGrid,
} from "lucide-react";
import type { Artifact } from "@/types";
import { Card, IconButton } from "@/components/ui";
import { cn, formatTime } from "@/lib/utils";
import { KPIGrid } from "@/components/artifacts/KPIGrid";
import { ChartArtifact } from "@/components/artifacts/ChartArtifact";
import { TableArtifact } from "@/components/artifacts/TableArtifact";
import { ValidationSummaryArtifact } from "@/components/artifacts/ValidationSummaryArtifact";

const KIND_ICON = {
  kpi: LayoutGrid,
  chart: BarChart3,
  table: Sheet,
  validation: ShieldCheck,
} as const;

function ArtifactBody({ artifact }: { artifact: Artifact }) {
  switch (artifact.data.kind) {
    case "kpi":
      return <KPIGrid data={artifact.data} />;
    case "chart":
      return <ChartArtifact data={artifact.data} />;
    case "table":
      return <TableArtifact data={artifact.data} />;
    case "validation":
      return <ValidationSummaryArtifact data={artifact.data} />;
  }
}

export function ArtifactCard({
  artifact,
  onTogglePin,
}: {
  artifact: Artifact;
  onTogglePin: (id: string) => void;
}) {
  const [collapsed, setCollapsed] = useState(false);
  const [copied, setCopied] = useState(false);
  const Icon = KIND_ICON[artifact.kind];

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(JSON.stringify(artifact.data, null, 2));
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable in some sandboxes */
    }
  };

  const download = () => {
    const blob = new Blob([JSON.stringify(artifact, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${artifact.id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <Card
      className={cn(
        "animate-fade-in overflow-hidden",
        artifact.pinned && "ring-1 ring-peri-400/30",
      )}
    >
      <div className="flex items-start gap-3 border-b border-[var(--color-line-soft)] px-4 py-3">
        <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-navy-700/60 text-peri-300">
          <Icon size={15} />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <h3 className="truncate text-sm font-semibold text-ink-100">{artifact.title}</h3>
            {artifact.pinned && (
              <Pin size={12} className="shrink-0 fill-peri-400 text-peri-400" />
            )}
          </div>
          <p className="mt-0.5 truncate text-xs text-ink-400">{artifact.description}</p>
          <div className="mt-1 flex items-center gap-2 text-[10px] font-medium uppercase tracking-wider text-ink-500">
            <span>{artifact.source}</span>
            <span className="h-0.5 w-0.5 rounded-full bg-ink-500" />
            <span>{formatTime(artifact.createdAt)}</span>
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-0.5">
          <IconButton
            label={artifact.pinned ? "Unpin" : "Pin"}
            active={artifact.pinned}
            onClick={() => onTogglePin(artifact.id)}
          >
            <Pin size={14} />
          </IconButton>
          <IconButton label={copied ? "Copied" : "Copy data"} onClick={copy}>
            {copied ? <Check size={14} className="text-mint-400" /> : <Copy size={14} />}
          </IconButton>
          <IconButton label="Download JSON" onClick={download}>
            <Download size={14} />
          </IconButton>
          <IconButton
            label={collapsed ? "Expand" : "Collapse"}
            onClick={() => setCollapsed((c) => !c)}
          >
            <ChevronDown
              size={14}
              className={cn("transition-transform", collapsed && "-rotate-90")}
            />
          </IconButton>
        </div>
      </div>
      {!collapsed && <div className="p-4">{<ArtifactBody artifact={artifact} />}</div>}
    </Card>
  );
}
