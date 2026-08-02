/**
 * Phase 2A — the completions movement hover.
 *
 * The same wrapper composition as `InsightLineChart`, over the origination
 * funnel's stage card. All state lives here; `FunnelStageCard` gained three
 * optional props and no logic, so the other three stage cards are untouched.
 *
 * The measure is a PIPELINE-STAGE one — the balance sitting at the COMPLETED
 * stage, whose week-on-week change is what the card's flow bars already plot.
 * The backend labels it "Completed case value" and states in its methodology
 * that it is not funded balance growth; nothing here relabels it.
 */

import { useCallback, useMemo, useState } from "react";
import type { AgentClient } from "@/api";
import type { FunnelConversion, FunnelFlowPoint, FunnelPoint,
  MovementDetailType, PipelineFunnelEvolution } from "@/domain";
import { FunnelStageCard } from "@/components/EvolutionPanel";
import { EnhancedMetricTooltip } from "./EnhancedMetricTooltip";
import { InsightDetailDrawer } from "./InsightDetailDrawer";
import { useMovementDetail } from "@/hooks/useMovementDetail";
import { formatGBP } from "@/lib/utils";

export interface InsightStageCardProps {
  stage: string;
  label: string;
  points: FunnelPoint[];
  flowPoints: FunnelFlowPoint[];
  summary: PipelineFunnelEvolution["summary"][string] | undefined;
  conversion: FunnelConversion | null;
  cohortPct: number | null;
  showCumulative: boolean;
  large?: boolean;
  onExpand?: () => void;
  enhanced?: boolean;
  latestWeek?: string | null;

  client: AgentClient;
  portfolioId: string;
  detailType: MovementDetailType;
  portfolioContext?: string;
  drillDown?: boolean;
  /**
   * Whether THIS stage gets the movement hover.
   *
   * Separate from `enhanced` (which controls the conversion evidence on every
   * stage) because the movement payload describes the COMPLETED stage only.
   * Showing it on KFI, Application or Offer would attach completions numbers to
   * a chart about something else.
   */
  movement?: boolean;
}

function BaseBody({ active, payload, label }: {
  active?: boolean;
  payload?: Array<{ name?: string; value?: number | string }>;
  label?: string | number;
}) {
  if (!active) return null;
  return (
    <div>
      <div className="text-[11px] text-ink-400">{String(label ?? "")}</div>
      {(payload ?? []).map((p, i) => (
        <div key={`${p.name}-${i}`} className="text-[12px] text-ink-100 tabular-nums">
          {p.name}: {typeof p.value === "number"
            ? formatGBP(p.value, { compact: true }) : String(p.value ?? "")}
        </div>
      ))}
    </div>
  );
}

export function InsightStageCard({
  client, portfolioId, detailType, portfolioContext, drillDown = false,
  movement = false, ...card
}: InsightStageCardProps) {
  const [active, setActive] = useState<string | null>(null);
  const [pinned, setPinned] = useState<string | null>(null);

  const state = useMovementDetail({
    client, portfolioId, detailType, portfolioContext,
    enabled: Boolean(card.enhanced && movement), asOf: pinned ?? active,
  });

  const renderBase = useCallback(
    (p: { active?: boolean; payload?: Array<{ name?: string; value?: number | string }>;
          label?: string | number }) => <BaseBody {...p} />, []);

  const tooltip = useMemo(
    () => (
      <EnhancedMetricTooltip renderBase={renderBase} detail={state.detail}
        loading={state.loading} unavailable={state.unavailable}
        showDrillHint={Boolean(drillDown && state.detail?.available)} />
    ),
    [renderBase, state.detail, state.loading, state.unavailable, drillDown],
  );

  // No movement hover for this stage -> the card renders exactly as before
  // (the `enhanced` conversion evidence is passed through untouched).
  if (!card.enhanced || !movement) return <FunnelStageCard {...card} />;

  return (
    <div data-testid="insight-stage-card">
      <FunnelStageCard {...card} tooltipContent={tooltip} onActivePoint={setActive}
        onPointClick={drillDown ? setPinned : undefined} />
      {pinned && (
        <InsightDetailDrawer detail={state.detail} loading={state.loading}
          onClose={() => setPinned(null)} />
      )}
    </div>
  );
}
