import { Suspense, lazy, useMemo } from "react";
import { AlertTriangle } from "lucide-react";
import type { ChartArtifact } from "@/domain";
import { applyTraktTheme } from "@/lib/plotlyTheme";

const PlotlyLazy = lazy(() => import("./PlotlyLazy"));

/** A figure is renderable when it's an object with at least one trace. */
export function hasPlotlyFigure(figure: unknown): figure is { data: unknown[]; layout?: Record<string, unknown> } {
  return (
    typeof figure === "object" &&
    figure !== null &&
    Array.isArray((figure as { data?: unknown }).data) &&
    (figure as { data: unknown[] }).data.length > 0
  );
}

function FigureError({ message }: { message: string }) {
  return (
    <div className="flex items-start gap-3 rounded-lg border border-rose-400/20 bg-rose-400/5 p-4">
      <AlertTriangle size={18} className="mt-0.5 shrink-0 text-rose-400" />
      <div>
        <div className="text-sm font-medium text-ink-100">Chart could not be rendered</div>
        <p className="mt-1 text-xs leading-relaxed text-ink-400">{message}</p>
      </div>
    </div>
  );
}

function Loading() {
  return (
    <div className="flex h-[320px] items-center justify-center text-xs font-medium text-ink-400">
      Loading chart
      <span className="ml-1 inline-flex gap-0.5">
        <span className="dot-1 h-1 w-1 rounded-full bg-peri-300" />
        <span className="dot-2 h-1 w-1 rounded-full bg-peri-300" />
        <span className="dot-3 h-1 w-1 rounded-full bg-peri-300" />
      </span>
    </div>
  );
}

/**
 * Fallback Plotly renderer. Used only when no native renderer can faithfully
 * represent a chart (see ArtifactRenderer). The backend figure is re-skinned by
 * `applyTraktTheme` (transparent background, Inter typography, navy/periwinkle
 * palette, soft gridlines, dark hover) so it matches the dashboard.
 */
export function PlotlyArtifactView({ artifact }: { artifact: ChartArtifact }) {
  const figure = artifact.source.figure;
  const themed = useMemo(
    () => (hasPlotlyFigure(figure) ? applyTraktTheme(figure) : null),
    [figure],
  );

  if (!themed) {
    return <FigureError message="The chart figure was missing or malformed." />;
  }
  return (
    <Suspense fallback={<Loading />}>
      <PlotlyLazy figure={themed} />
    </Suspense>
  );
}
