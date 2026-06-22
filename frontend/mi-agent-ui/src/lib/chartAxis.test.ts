import { describe, expect, it } from "vitest";
import { paddedDomain } from "./chartAxis";

describe("paddedDomain — population-aware axis framing", () => {
  it("frames a borrower-age population without wasting area below it", () => {
    // Population starts at 55 — the axis must not start at 0.
    const rows = [{ age: 55 }, { age: 62 }, { age: 71 }, { age: 88 }];
    const d = paddedDomain(rows, "age")!;
    expect(d.domain[0]).toBeGreaterThan(40);
    expect(d.domain[0]).toBeLessThanOrEqual(55);
    expect(d.domain[1]).toBeGreaterThanOrEqual(88);
  });

  it("snaps a fraction-stored LTV axis to 10-point percent ticks", () => {
    // LTV stored as 0..1 fraction; population ~ 32%–58%.
    const rows = [{ ltv: 0.32 }, { ltv: 0.45 }, { ltv: 0.58 }];
    const d = paddedDomain(rows, "ltv", { isPercent: true, scale: "percent_fraction" })!;
    // Domain returned in DATA (fraction) units.
    expect(d.domain[0]).toBeGreaterThanOrEqual(0.2);
    expect(d.domain[0]).toBeLessThanOrEqual(0.32);
    expect(d.domain[1]).toBeGreaterThanOrEqual(0.58);
    // Ticks land on whole 10% boundaries (0.2, 0.3, 0.4, …).
    expect(d.ticks).toBeDefined();
    for (const t of d.ticks!) {
      expect(Math.round(t * 1000) % 100).toBe(0);
    }
    expect(d.ticks!.map((t) => Math.round(t * 100))).toContain(30);
    expect(d.ticks!.map((t) => Math.round(t * 100))).toContain(40);
  });

  it("never lets a percent axis go below zero", () => {
    const rows = [{ ltv: 4 }, { ltv: 8 }];
    const d = paddedDomain(rows, "ltv", { isPercent: true, scale: "percent_points" })!;
    expect(d.domain[0]).toBe(0);
  });

  it("returns null when there is nothing to frame", () => {
    expect(paddedDomain([], "age")).toBeNull();
    expect(paddedDomain([{ age: 1 }], undefined)).toBeNull();
  });
});
