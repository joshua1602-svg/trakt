/**
 * The additivity + population contract, and the failure it exists to prevent.
 *
 * THE AUDIT CASE, exactly. Asked "average balance by broker channel" over a book
 * with thirteen brokers, the engine classifies the measure NON-ADDITIVE, caps
 * the chart to ten and drops the tail. React used to decide additivity from the
 * DISPLAY FORMAT — `gbp` meant sum-able — so it summed ten broker AVERAGES into
 * a "portfolio total" of £3,060,094 against a real funded book of £38,646,184,
 * and the Insight Panel told the user:
 *
 *     "Broker E has the largest current outstanding balance avg, at 11% of the total."
 *     "The top 3 account for 31% of current outstanding balance avg."
 *
 * The fixtures below are the REAL artifacts from POST /mi/query — captured from
 * the route, not hand-written — so these tests fail the moment either the
 * engine stops publishing the contract or React starts guessing again.
 *
 * See docs/reports/mi_cross_channel_ownership_audit.md.
 */

import { describe, expect, it } from "vitest";
import fixtures from "@/test/fixtures/mi_query_broker_artifacts.json";
import { buildDrillModel } from "@/lib/drill";
import { computeInsights } from "@/lib/insights";

/* eslint-disable @typescript-eslint/no-explicit-any */
const AVG: any = (fixtures as any).avg.artifact;
const SUM: any = (fixtures as any).sum.artifact;

const measureKeyOf = (a: any) => a.series[0].key as string;

// --------------------------------------------------------------------------
// The contract the engine publishes.
// --------------------------------------------------------------------------

describe("the engine publishes additivity and population completeness", () => {
  it("marks a monetary AVERAGE non-additive", () => {
    const key = measureKeyOf(AVG);
    expect(key).toContain("_avg");
    expect(AVG.displayHints[key].format).toBe("gbp");     // money…
    expect(AVG.displayHints[key].additive).toBe(false);   // …and NOT sum-able
  });

  it("marks a monetary SUM additive", () => {
    const key = measureKeyOf(SUM);
    expect(SUM.displayHints[key].format).toBe("gbp");
    expect(SUM.displayHints[key].additive).toBe(true);
  });

  it("flags the truncated population, and whether the value survived it", () => {
    // Both were capped 13 -> 10. Only the additive one keeps an "Other" bucket,
    // so only the additive one still represents the whole.
    expect(AVG.population).toMatchObject({
      returnedCount: 10, totalCount: 13, truncated: true, populationComplete: false,
    });
    expect(SUM.population).toMatchObject({
      returnedCount: 10, totalCount: 13, truncated: true, populationComplete: true,
    });
  });
});

// --------------------------------------------------------------------------
// React consumes it, and never re-derives it.
// --------------------------------------------------------------------------

describe("React reads additivity from the contract", () => {
  it("does not treat a money-formatted average as additive", () => {
    const model = buildDrillModel(AVG);
    const measure = model!.measures.find((m) => m.key === measureKeyOf(AVG))!;
    expect(measure.format).toBe("gbp");
    expect(measure.additive).toBe(false);
    expect(model!.primary).toBeUndefined();     // nothing to sum
    expect(model!.totals[measureKeyOf(AVG)]).toBeUndefined();
  });

  it("still treats a money-formatted sum as additive", () => {
    const model = buildDrillModel(SUM);
    const measure = model!.measures.find((m) => m.key === measureKeyOf(SUM))!;
    expect(measure.additive).toBe(true);
    expect(model!.primary?.key).toBe(measureKeyOf(SUM));
  });

  it("format alone cannot make a measure additive", () => {
    // The old rule: format ∈ {gbp, number} ⇒ additive. Strip the contract and
    // the measure must fall back to NOT additive, never to the format.
    const stripped = {
      ...AVG,
      displayHints: { ...AVG.displayHints, [measureKeyOf(AVG)]: { format: "gbp", scale: null } },
    };
    const model = buildDrillModel(stripped);
    expect(model!.measures[0].additive).toBe(false);
  });
});

// --------------------------------------------------------------------------
// THE ORIGINAL FAILURE.
// --------------------------------------------------------------------------

describe("the broker-average failure", () => {
  it("produces no fake total and no share of one", () => {
    const out = computeInsights(AVG, { metric: measureKeyOf(AVG) } as any);
    expect(out?.statistics.topShare).toBeUndefined();
    expect(out?.statistics.top3Share).toBeUndefined();
  });

  it("states no part-to-whole claim in any observation", () => {
    const out = computeInsights(AVG, { metric: measureKeyOf(AVG) } as any);
    const text = (out?.observations ?? []).map((o) => o.text).join(" | ");
    expect(text).not.toMatch(/of the total/i);
    expect(text).not.toMatch(/account for/i);
    expect(text).not.toMatch(/\d+%/);
  });

  it("keeps the legitimate non-share observations", () => {
    // Suppression must be surgical: a range statement over ten averages is
    // still true and still useful.
    const out = computeInsights(AVG, { metric: measureKeyOf(AVG) } as any);
    expect((out?.observations ?? []).length).toBeGreaterThan(0);
  });
});

describe("an additive ranked result keeps its correct shares", () => {
  it("computes a share, because the Other bucket preserves the whole", () => {
    const out = computeInsights(SUM, { metric: measureKeyOf(SUM) } as any);
    expect(out?.statistics.topShare).toBeGreaterThan(0);
    expect(out?.statistics.topShare).toBeLessThanOrEqual(1);
    const rowsTotal = SUM.rows.reduce(
      (acc: number, r: any) => acc + Number(r[measureKeyOf(SUM)] ?? 0), 0);
    expect(out?.statistics.total).toBeCloseTo(rowsTotal, 2);
  });
});

describe("an incomplete population suppresses shares even when additive", () => {
  it("refuses a share when the engine says the rows are not the whole", () => {
    const partial = { ...SUM, population: { ...SUM.population, populationComplete: false } };
    const out = computeInsights(partial, { metric: measureKeyOf(SUM) } as any);
    expect(out?.statistics.topShare).toBeUndefined();
  });

  it("allows a share when no population contract is present", () => {
    // An older payload keeps working: the additive + snapshot guards carry it.
    const { population: _drop, ...noContract } = SUM;
    const out = computeInsights(noContract as any, { metric: measureKeyOf(SUM) } as any);
    expect(out?.statistics.topShare).toBeGreaterThan(0);
  });
});
