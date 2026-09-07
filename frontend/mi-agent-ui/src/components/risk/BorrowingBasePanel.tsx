/**
 * Borrowing Base — the facility position, at the top of Eligibility &
 * Concentrations.
 *
 * Every figure is rendered from the governed envelope verbatim. The panel does
 * no arithmetic: it does not divide, does not cap, does not floor a negative
 * headroom, and does not convert a missing input into a zero. A measure that
 * arrives as `NOT_CALCULABLE` is shown as an explicit governed status with the
 * missing input named — never as "—", which reads as "nothing to report".
 *
 * The one presentation-only liberty, and it is stated on the tile: when the
 * facility is OVER-DRAWN the headroom tile shows £0 and a separate Deficiency
 * tile carries the shortfall. The governed number stays negative underneath;
 * this is how an operator reads it, not what the calculation says.
 */

import type {
  BorrowingBaseSnapshot,
  EligiblePopulationDisclosure,
  Measure,
} from "@/domain";
import { NOT_CALCULABLE } from "@/domain";
import { Badge, Card } from "@/components/ui";
import { formatMoney, formatPct } from "@/lib/utils";

/** A human phrase for a missing input, so the tile says WHAT is missing. */
const MISSING_INPUT_LABEL: Record<string, string> = {
  current_drawn_amount: "facility drawings not supplied",
  facility_commitment: "commitment not configured",
  advance_rate: "advance rate not configured",
  concentration_denominator_floor: "denominator floor not configured",
};

function isNumber(value: Measure | undefined): value is number {
  return typeof value === "number";
}

function money(value: Measure | undefined): string | null {
  return isNumber(value) ? formatMoney(value) : null;
}

function KpiTile({
  label,
  value,
  sublabel,
  tone = "neutral",
  status,
  testId,
}: {
  label: string;
  value: string | null;
  sublabel?: string;
  tone?: "mint" | "amber" | "rose" | "neutral";
  /** Shown INSTEAD of a number when the measure is not calculable. */
  status?: string | null;
  testId?: string;
}) {
  const toneClass =
    tone === "rose"
      ? "text-rose-400"
      : tone === "amber"
        ? "text-amber-400"
        : tone === "mint"
          ? "text-mint-400"
          : "text-ink-100";
  return (
    <div
      data-testid={testId}
      className="rounded-lg border border-[var(--color-line-soft)] bg-navy-900/50 px-3 py-2"
    >
      <p className="text-[10px] uppercase tracking-wider text-ink-500">{label}</p>
      {value !== null ? (
        <p className={`font-mono text-[18px] tabular-nums ${toneClass}`}>{value}</p>
      ) : (
        <p className="pt-0.5">
          <Badge tone="neutral">
            <span aria-hidden>–</span>
            Not calculable
          </Badge>
        </p>
      )}
      {(sublabel || status) && (
        <p className="mt-0.5 text-[9px] leading-tight text-ink-500">
          {value !== null ? sublabel : (status ?? sublabel)}
        </p>
      )}
    </div>
  );
}

function EligibilityRow({
  label,
  count,
  balance,
  share,
  tone,
  testId,
}: {
  label: string;
  count: number | undefined;
  balance: number | undefined;
  share: number | null | undefined;
  tone: "mint" | "amber" | "neutral";
  testId: string;
}) {
  const toneClass =
    tone === "mint" ? "text-mint-400" : tone === "amber" ? "text-amber-400" : "text-ink-300";
  return (
    <div
      data-testid={testId}
      className="grid grid-cols-[1fr_auto_auto_auto] items-baseline gap-3 border-b border-[var(--color-line-soft)] px-1 py-1.5 text-[12px] last:border-0"
    >
      <span className={toneClass}>{label}</span>
      <span className="text-right font-mono tabular-nums text-ink-300">
        {count === undefined ? "—" : count.toLocaleString("en-GB")}
      </span>
      <span className="text-right font-mono tabular-nums text-ink-200">
        {balance === undefined ? "—" : formatMoney(balance)}
      </span>
      <span className="w-16 text-right font-mono tabular-nums text-ink-400">
        {share === null || share === undefined ? "—" : formatPct(share, 1)}
      </span>
    </div>
  );
}

export function BorrowingBasePanel({
  snapshot,
  population,
  onShowLoans,
}: {
  snapshot: BorrowingBaseSnapshot | undefined;
  population?: EligiblePopulationDisclosure;
  /** Open the existing loan drill-down for one eligibility status. */
  onShowLoans?: (status: "ELIGIBLE" | "INELIGIBLE" | "UNDETERMINED") => void;
}) {
  if (!snapshot) return null;

  if (!snapshot.available) {
    return (
      <Card className="p-3 space-y-1.5" testId="borrowing-base-panel">
        <h3 className="text-[12px] font-semibold text-ink-100">Borrowing base</h3>
        <p
          role="note"
          data-testid="borrowing-base-unavailable"
          className="rounded-lg border border-[var(--color-line-soft)] bg-navy-900/50 px-3 py-2 text-[11px] text-ink-400"
        >
          {snapshot.reason ??
            "No funding facility is configured for this portfolio."}
        </p>
      </Card>
    );
  }

  const facility = snapshot.facility;
  const missing = new Set(snapshot.missingInputs ?? []);
  const missingPhrase = (key: string) =>
    missing.has(key) ? (MISSING_INPUT_LABEL[key] ?? `${key} not supplied`) : null;

  const headroom = snapshot.borrowingBaseHeadroom;
  const deficiency = snapshot.borrowingBaseDeficiency;
  // Presentation only: an over-drawn facility shows £0 headroom beside the
  // deficiency. The governed value stays negative in `snapshot`.
  const overDrawn = isNumber(headroom) && headroom < 0;
  const headroomDisplay = overDrawn ? formatMoney(0) : money(headroom);

  const utilisation = snapshot.facilityUtilisationPct;
  const utilisationTone = !isNumber(utilisation)
    ? "neutral"
    : utilisation >= 100
      ? "rose"
      : utilisation >= 90
        ? "amber"
        : "mint";

  const denominatorFloorBinding = snapshot.concentrationDenominatorFloorBinding;
  const prototype = (snapshot.prototypeAssumptionsUsed ?? []).length > 0;
  const unreconciled = snapshot.reconciles === false;

  return (
    <Card className="space-y-2 p-3" testId="borrowing-base-panel">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-[12px] font-semibold text-ink-100">
          Borrowing base
          {facility && (
            <span className="ml-2 font-normal text-[11px] text-ink-500">
              {facility.facilityLabel} · {facility.facilityType.replace(/_/g, " ")}
            </span>
          )}
        </h3>
        <p className="text-[10px] text-ink-500">
          {isNumber(snapshot.advanceRatePct) && (
            <>Advance rate {formatPct(snapshot.advanceRatePct, 0)}</>
          )}
          {isNumber(snapshot.facilityCommitment) && (
            <> · Commitment {formatMoney(snapshot.facilityCommitment)}</>
          )}
        </p>
      </div>

      {unreconciled && (
        <p
          role="alert"
          data-testid="borrowing-base-reconciliation-failed"
          className="rounded-lg border border-rose-400/30 bg-rose-400/5 px-3 py-2 text-[11px] text-rose-300"
        >
          The eligibility population does not reconcile, so these figures are
          shown as diagnostics and are NOT a governed borrowing base.
        </p>
      )}

      {prototype && (
        <p
          role="note"
          data-testid="borrowing-base-prototype-banner"
          className="rounded-lg border border-amber-400/20 bg-amber-400/5 px-3 py-2 text-[11px] text-amber-300/90"
        >
          <span className="font-semibold">Prototype assumption in use. </span>
          {snapshot.prototypeAssumptionsUsed?.join(" ")}
        </p>
      )}

      {/* KPIs */}
      <div
        className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5"
        data-testid="borrowing-base-kpis"
      >
        <KpiTile
          label="Eligible collateral"
          value={money(snapshot.eligibleCurrentBalance)}
          sublabel={`${(snapshot.eligibleLoanCount ?? 0).toLocaleString("en-GB")} loans`}
          tone="neutral"
          testId="bb-eligible-collateral"
        />
        <KpiTile
          label="Borrowing base"
          value={money(snapshot.availableBorrowingBase)}
          sublabel={
            snapshot.facilityCapBinding
              ? "capped at facility commitment"
              : isNumber(snapshot.grossBorrowingBase)
                ? `gross ${formatMoney(snapshot.grossBorrowingBase)}`
                : undefined
          }
          status={missingPhrase("advance_rate") ?? undefined}
          tone="neutral"
          testId="bb-borrowing-base"
        />
        <KpiTile
          label="Facility drawn"
          value={money(snapshot.currentDrawnAmount)}
          status={missingPhrase("current_drawn_amount") ?? undefined}
          sublabel={facility?.currentDrawnAmountAsOf ?? undefined}
          tone="neutral"
          testId="bb-facility-drawn"
        />
        <KpiTile
          label="Headroom"
          value={headroomDisplay}
          status={missingPhrase("current_drawn_amount") ?? undefined}
          sublabel={
            overDrawn && isNumber(deficiency)
              ? `over-drawn — deficiency ${formatMoney(deficiency)}`
              : undefined
          }
          tone={overDrawn ? "rose" : "mint"}
          testId="bb-headroom"
        />
        <KpiTile
          label="Facility utilisation"
          value={isNumber(utilisation) ? formatPct(utilisation, 1) : null}
          status={missingPhrase("current_drawn_amount") ?? undefined}
          sublabel={
            isNumber(snapshot.borrowingBaseUtilisationPct)
              ? `${formatPct(snapshot.borrowingBaseUtilisationPct, 1)} of borrowing base`
              : undefined
          }
          tone={utilisationTone}
          testId="bb-facility-utilisation"
        />
      </div>

      {overDrawn && isNumber(deficiency) && (
        <p
          role="alert"
          data-testid="borrowing-base-deficiency"
          className="rounded-lg border border-rose-400/30 bg-rose-400/5 px-3 py-2 text-[11px] text-rose-300"
        >
          Borrowing-base deficiency {formatMoney(deficiency)} — drawings exceed
          the available borrowing base.
        </p>
      )}

      {/* Eligibility split */}
      <div className="rounded-lg border border-[var(--color-line-soft)] bg-navy-900/40 p-2">
        <div className="grid grid-cols-[1fr_auto_auto_auto] gap-3 border-b border-[var(--color-line-soft)] px-1 pb-1 text-[10px] uppercase tracking-wider text-ink-500">
          <span>Financing Portfolio</span>
          <span className="text-right">Loans</span>
          <span className="text-right">Current balance</span>
          <span className="w-16 text-right">% of book</span>
        </div>
        <EligibilityRow
          label="Eligible"
          count={snapshot.eligibleLoanCount}
          balance={snapshot.eligibleCurrentBalance}
          share={snapshot.eligibleShareOfFinancingPortfolioPct}
          tone="mint"
          testId="bb-split-eligible"
        />
        <EligibilityRow
          label="Ineligible"
          count={snapshot.ineligibleLoanCount}
          balance={snapshot.ineligibleCurrentBalance}
          share={snapshot.ineligibleShareOfFinancingPortfolioPct}
          tone="neutral"
          testId="bb-split-ineligible"
        />
        <EligibilityRow
          label="Undetermined"
          count={snapshot.undeterminedLoanCount}
          balance={snapshot.undeterminedCurrentBalance}
          share={snapshot.undeterminedShareOfFinancingPortfolioPct}
          tone="amber"
          testId="bb-split-undetermined"
        />
        <div className="flex flex-wrap items-center justify-between gap-2 px-1 pt-1.5">
          <p className="text-[10px] text-ink-500">
            Financing Portfolio{" "}
            {isNumber(snapshot.financingPortfolioBalance)
              ? formatMoney(snapshot.financingPortfolioBalance)
              : "—"}{" "}
            over {(snapshot.financingPortfolioLoanCount ?? 0).toLocaleString("en-GB")} loans
            {isNumber(snapshot.concentrationLimitDenominator) && (
              <>
                {" "}
                · Concentration Limit Denominator{" "}
                {formatMoney(snapshot.concentrationLimitDenominator)}
                {denominatorFloorBinding ? " (contractual floor binding)" : ""}
              </>
            )}
          </p>
          {onShowLoans && (
            <span className="flex gap-2 text-[11px]">
              {(["ELIGIBLE", "INELIGIBLE", "UNDETERMINED"] as const).map((s) => (
                <button
                  key={s}
                  type="button"
                  className="text-cyan-200 underline-offset-2 hover:underline"
                  onClick={() => onShowLoans(s)}
                >
                  Show {s.toLowerCase()} loans
                </button>
              ))}
            </span>
          )}
        </div>
      </div>

      {/* Binding concentration + the treatment of breaches */}
      <p className="px-1 text-[10px] leading-relaxed text-ink-500">
        {snapshot.nearestConcentrationLimit &&
        snapshot.nearestConcentrationLimit !== NOT_CALCULABLE ? (
          <>
            Closest concentration limit: {snapshot.nearestConcentrationLimit}
            {isNumber(snapshot.nearestConcentrationHeadroomPct) && (
              <> · {formatPct(snapshot.nearestConcentrationHeadroomPct, 2)} headroom</>
            )}
            {isNumber(snapshot.nearestConcentrationHeadroomAmount) && (
              <> ({formatMoney(snapshot.nearestConcentrationHeadroomAmount)})</>
            )}
            {(snapshot.breachedConcentrationCount ?? 0) > 0 && (
              <span className="text-rose-300">
                {" "}
                · {snapshot.breachedConcentrationCount} breached
              </span>
            )}
            .{" "}
          </>
        ) : null}
        {snapshot.concentrationAdjustment?.note}
        {population && population.basis !== "governed_eligibility" && (
          <span className="block text-amber-300/80">{population.note}</span>
        )}
      </p>
    </Card>
  );
}
