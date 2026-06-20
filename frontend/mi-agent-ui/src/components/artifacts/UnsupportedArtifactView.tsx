import { FileWarning } from "lucide-react";
import type { UnsupportedArtifact } from "@/domain";

/**
 * Explicit "received but not natively rendered" state. Shows the reason and,
 * when a raw Plotly figure was preserved upstream, notes that it is available
 * for a future Plotly renderer (we do not bundle plotly.js today).
 */
export function UnsupportedArtifactView({ artifact }: { artifact: UnsupportedArtifact }) {
  const hasFigure = Boolean(artifact.source.figure);
  return (
    <div className="flex items-start gap-3 rounded-lg border border-amber-400/20 bg-amber-400/5 p-4">
      <FileWarning size={18} className="mt-0.5 shrink-0 text-amber-400" />
      <div className="min-w-0">
        <div className="text-sm font-medium text-ink-100">Not rendered in this view</div>
        <p className="mt-1 text-xs leading-relaxed text-ink-400">{artifact.reason}</p>
        {hasFigure && (
          <p className="mt-1.5 text-[11px] text-ink-500">
            A raw Plotly figure was preserved in this artifact's metadata and can be
            rendered once a Plotly view is added.
          </p>
        )}
      </div>
    </div>
  );
}
