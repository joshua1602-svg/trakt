import type { PipelineMovement, PipelineStageMovement } from "../../domain/evolution";
import { formatGBP } from "@/lib/utils";

const STAGE_LABELS: Record<string, string> = {
  KFI: "KFI",
  APPLICATION: "Application",
  OFFER: "Offer",
  COMPLETED: "Completed",
  WITHDRAWN: "Withdrawn",
  ABSENT: "Left the extract",
};

function stageLabel(stage: string): string {
  const key = String(stage || "").toUpperCase();
  return STAGE_LABELS[key] ?? key.charAt(0) + key.slice(1).toLowerCase();
}

/** A leg of the identity. A zero leg is a dash — "−0  £0" reads as a fault. */
function Leg({ count, amount, sign }: {
  count: number; amount: number; sign?: "+" | "−";
}) {
  if (!count && !amount) return <span className="text-ink-500">—</span>;
  return (
    <span className="tabular-nums">
      <span className="text-ink-200">{sign}{count.toLocaleString()}</span>
      <span className="ml-2 text-ink-400">{formatGBP(amount)}</span>
    </span>
  );
}

/**
 * What happened to pipeline cases between two governed weekly extracts.
 *
 * Per live stage, on counts AND amounts:
 *
 *   opening live + arrivals − departures ± amount change on stayers
 *     = closing live
 *
 * Every figure comes from `/mi/evolution/pipeline-movement`, which is the same
 * computation the investor pack renders — the renderer decides presentation,
 * the engine decides truth. Nothing is derived here.
 */
export function StageMovementPanel({ movement }: { movement: PipelineMovement | null }) {
  if (!movement) return null;

  if (!movement.available) {
    return (
      <div className="rounded-xl border border-[var(--color-line)] bg-navy-900/40 p-4"
        data-testid="stage-movement-unavailable">
        <div className="text-[12px] font-semibold text-ink-200">
          Pipeline stage movement
        </div>
        <p className="mt-2 text-[11px] leading-relaxed text-ink-400">
          {movement.reason
            ?? "Case-level stage movement is not available for this portfolio."}
        </p>
      </div>
    );
  }

  const stages = movement.stages ?? [];
  const departures = new Map<string, { cases: number; amount: number }>();
  for (const stage of stages) {
    for (const dest of stage.departuresByDestination ?? []) {
      const key = String(dest.stage || "").toUpperCase();
      const bucket = departures.get(key) ?? { cases: 0, amount: 0 };
      bucket.cases += dest.caseCount ?? 0;
      bucket.amount += dest.amount ?? 0;
      departures.set(key, bucket);
    }
  }
  const destinations = [...departures.entries()]
    .filter(([, v]) => v.amount || v.cases)
    .sort((a, b) => b[1].amount - a[1].amount);

  return (
    <div className="rounded-xl border border-[var(--color-line)] bg-navy-900/40 p-4 lg:col-span-2"
      data-testid="stage-movement">
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-[12px] font-semibold text-ink-200">
          Pipeline stage movement
        </div>
        <div className="text-[10px] text-ink-500">
          {movement.openingWeek} → {movement.closingWeek}
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full min-w-[42rem] text-[11px]">
          <thead>
            <tr className="border-b border-[var(--color-line)] text-left text-ink-400">
              <th className="py-1 pr-3 font-medium">Stage</th>
              <th className="py-1 pr-3 text-right font-medium">Opening</th>
              <th className="py-1 pr-3 text-right font-medium">Arrived</th>
              <th className="py-1 pr-3 text-right font-medium">Departed</th>
              <th className="py-1 pr-3 text-right font-medium">On stayers</th>
              <th className="py-1 text-right font-medium">Closing</th>
            </tr>
          </thead>
          <tbody>
            {stages.map((stage: PipelineStageMovement) => (
              <tr key={stage.stage} className="border-b border-[var(--color-line)]/40">
                <td className="py-2 pr-3 text-ink-200">{stageLabel(stage.stage)}</td>
                <td className="py-2 pr-3 text-right">
                  <Leg count={stage.openingCaseCount} amount={stage.openingAmount} />
                </td>
                <td className="py-2 pr-3 text-right">
                  <Leg count={stage.arrivalCaseCount} amount={stage.arrivalAmount} sign="+" />
                </td>
                <td className="py-2 pr-3 text-right">
                  <Leg count={stage.departureCaseCount} amount={stage.departureAmount} sign="−" />
                </td>
                <td className="py-2 pr-3 text-right tabular-nums text-ink-400">
                  {formatGBP(stage.amountChangeOnPersisting)}
                </td>
                <td className="py-2 text-right">
                  <Leg count={stage.closingCaseCount} amount={stage.closingAmount} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {destinations.length > 0 ? (
        <div className="mt-4" data-testid="stage-movement-destinations">
          <div className="mb-2 text-[11px] font-semibold text-ink-300">
            Where departing cases went
          </div>
          <ul className="space-y-1">
            {destinations.map(([key, value]) => (
              <li key={key} className="flex items-baseline justify-between gap-3 text-[11px]">
                <span className="text-ink-400">{stageLabel(key)}</span>
                <span className="tabular-nums text-ink-200">
                  {value.cases.toLocaleString()} · {formatGBP(value.amount)}
                </span>
              </li>
            ))}
          </ul>
        </div>
      ) : (
        <p className="mt-3 text-[11px] italic text-ink-500">
          No case left a live stage between these two extracts.
        </p>
      )}

      <p className="mt-3 text-[10px] leading-relaxed text-ink-500">
        Reconciled on the governed case identifier
        {movement.identifierField ? ` (${movement.identifierField})` : ""}: opening
        live + arrivals − departures ± amount change on persisting cases = closing
        live. A case that left a stage has not necessarily left the pipeline, and
        an amount amendment is a movement on the same case rather than an exit and
        an arrival.
      </p>
    </div>
  );
}
