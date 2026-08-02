/**
 * Phase 2A — wrapper composition around the existing line chart.
 *
 * All of the hover/drill state lives HERE, not inside `EvoLineChart`: the chart
 * gained three optional props and no logic, so every other chart that renders
 * through it is untouched and this whole layer can be deleted by removing one
 * directory and one call site.
 *
 * When `enabled` is false this renders `EvoLineChart` with no extra props at
 * all — i.e. the current production chart, with the current production tooltip.
 */

import { useCallback, useMemo, useState } from "react";
import type { AgentClient } from "@/api";
import type { MovementDetailType } from "@/domain";
import { EvoLineChart } from "@/components/EvolutionPanel";
import { EnhancedMetricTooltip } from "./EnhancedMetricTooltip";
import { useMovementDetail } from "@/hooks/useMovementDetail";
import { formatGBP } from "@/lib/utils";

export interface InsightLineChartProps {
  title: string;
  data: Array<Record<string, number | string | null>>;
  lines: { key: string; label: string }[];
  valueFormat?: "gbp" | "count" | "pct" | "pct_points";
  source?: string | null;
  asOf?: string | null;

  enabled: boolean;
  client: AgentClient;
  portfolioId: string;
  detailType: MovementDetailType;
  portfolioContext?: string;
  /** Opens the drill panel for a week. Absent until the drawer is wired up. */
  onOpenDetail?: (week: string) => void;
}

/** Recharts hands the tooltip `{active, payload, label}`; this renders the same
 * body the default tooltip would, so the enhanced tooltip is strictly additive
 * rather than a re-implementation of the chart's own formatting. */
function BaseBody({ active, payload, label, format }: {
  active?: boolean;
  payload?: Array<{ name?: string; value?: number | string }>;
  label?: string | number;
  format: (v: number) => string;
}) {
  if (!active) return null;
  return (
    <div>
      <div className="text-[11px] text-ink-400">{String(label ?? "")}</div>
      {(payload ?? []).map((p, i) => (
        <div key={`${p.name}-${i}`} className="text-[12px] text-ink-100 tabular-nums">
          {p.name}: {typeof p.value === "number" ? format(p.value) : String(p.value ?? "")}
        </div>
      ))}
    </div>
  );
}

export function InsightLineChart({
  enabled, client, portfolioId, detailType, portfolioContext, onOpenDetail,
  ...chart
}: InsightLineChartProps) {
  const [active, setActive] = useState<string | null>(null);

  const state = useMovementDetail({
    client, portfolioId, detailType, portfolioContext, enabled, asOf: active,
  });

  const format = useCallback(
    (v: number) => (chart.valueFormat === "count"
      ? v.toLocaleString("en-GB")
      : formatGBP(v, { compact: true })),
    [chart.valueFormat],
  );

  // Memoised on the state it actually reads, so the chart's `memo` still holds
  // between pointer moves that change nothing.
  const tooltip = useMemo(
    () => (
      <EnhancedMetricTooltip
        renderBase={(p) => <BaseBody {...p} format={format} />}
        detail={state.detail}
        loading={state.loading}
        unavailable={state.unavailable}
        showDrillHint={Boolean(onOpenDetail && state.detail?.available)}
      />
    ),
    [format, state.detail, state.loading, state.unavailable, onOpenDetail],
  );

  if (!enabled) return <EvoLineChart {...chart} />;

  return (
    <div
      data-testid="insight-line-chart"
      // Hover must never be the only route to this information: the wrapper is
      // focusable and Enter/Space opens the same panel a click does.
      tabIndex={onOpenDetail ? 0 : -1}
      role={onOpenDetail ? "button" : undefined}
      aria-label={onOpenDetail
        ? `${chart.title} — open movement details`
        : undefined}
      onKeyDown={(e) => {
        if (!onOpenDetail) return;
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          const week = active ?? lastWeek(chart.data);
          if (week) onOpenDetail(week);
        }
      }}
    >
      <EvoLineChart
        {...chart}
        tooltipContent={tooltip}
        onActivePoint={setActive}
        onPointClick={onOpenDetail}
      />
    </div>
  );
}

/** The most recent point — what a keyboard user means by "the latest week". */
function lastWeek(data: Array<Record<string, number | string | null>>): string | null {
  const last = data[data.length - 1];
  return last && last.period != null ? String(last.period) : null;
}
