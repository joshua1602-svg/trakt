import { Suspense, lazy } from "react";
import { AlertTriangle } from "lucide-react";
import type { ChartArtifact } from "@/domain";
import type { PlotlyFigure } from "./PlotlyLazy";

const PlotlyLazy = lazy(() => import("./PlotlyLazy"));

/** A figure is renderable when it's an object with at least one trace. */
export function hasPlotlyFigure(figure: unknown): figure is PlotlyFigure {
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
    <div className="flex h-[320px] items-center justify-center rounded-lg bg-white/95">
      <span className="inline-flex items-center gap-1.5 text-xs font-medium text-navy-700">
        Loading chart
        <span className="inline-flex gap-0.5">
          <span className="dot-1 h-1 w-1 rounded-full bg-navy-700" />
          <span className="dot-2 h-1 w-1 rounded-full bg-navy-700" />
          <span className="dot-3 h-1 w-1 rounded-full bg-navy-700" />
        </span>
      </span>
    </div>
  );
}

/**
 * Renders the backend-native Plotly figure (`source.figure`) faithfully. The
 * figure carries the chart factory's own theme (white background), so we wrap
 * it in a light panel that sits cleanly inside the dark ArtifactCard.
 */
export function PlotlyArtifactView({ artifact }: { artifact: ChartArtifact }) {
  const figure = artifact.source.figure;
  if (!hasPlotlyFigure(figure)) {
    return <FigureError message="The chart figure was missing or malformed." />;
  }
  return (
    <div className="overflow-hidden rounded-lg bg-white p-2">
      <Suspense fallback={<Loading />}>
        <PlotlyLazy figure={figure} />
      </Suspense>
    </div>
  );
}
