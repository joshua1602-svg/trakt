import { describe, expect, it } from "vitest";
import { bucketBound, cleanBucketLabel, sortStratBars } from "./stratOrder";

const labels = <T extends { label: string }>(bars: T[]) => bars.map((b) => b.label);
const bars = (...ls: string[]) => ls.map((label, i) => ({ label, value: 100 - i }));

describe("cleanBucketLabel", () => {
  it("strips a redundant .0 fraction from vintage years", () => {
    expect(cleanBucketLabel("2008.0")).toBe("2008");
    expect(cleanBucketLabel("1990.0")).toBe("1990");
  });
  it("leaves genuine decimals and ranges alone", () => {
    expect(cleanBucketLabel("6.5%")).toBe("6.5%");
    expect(cleanBucketLabel("40-50%")).toBe("40-50%");
  });
});

describe("bucketBound", () => {
  it("parses range, open and boundary shapes", () => {
    expect(bucketBound("40-50%")).toBe(40);
    expect(bucketBound("40–50%")).toBe(40); // en dash
    expect(bucketBound("85+")).toBe(85);
    expect(bucketBound(">=100%")).toBe(100);
    expect(bucketBound("£100K-£150K")).toBe(100);
    expect(bucketBound("2008.0")).toBe(2008);
  });
  it("sorts a below-boundary bucket ahead of its boundary", () => {
    expect(bucketBound("<20%")!).toBeLessThan(bucketBound("20-30%")!);
  });
  it("returns null for unknown-style and non-numeric labels", () => {
    expect(bucketBound("Unknown")).toBeNull();
    expect(bucketBound("London")).toBeNull();
  });
});

describe("sortStratBars", () => {
  it("orders LTV bands by band, not by balance", () => {
    const sorted = sortStratBars(bars("40-50%", "50-60%", "30-40%", "60-70%", "<20%", ">=100%"));
    expect(labels(sorted)).toEqual(["<20%", "30-40%", "40-50%", "50-60%", "60-70%", ">=100%"]);
  });

  it("orders age bands ascending with open-ended and Unknown buckets placed sensibly", () => {
    const sorted = sortStratBars(bars("85+", "75-80", "80-85", "Unknown"));
    expect(labels(sorted)).toEqual(["75-80", "80-85", "85+", "Unknown"]);
  });

  it("orders vintage years ascending and cleans float-cast labels via cleanBucketLabel", () => {
    const sorted = sortStratBars(bars("2008.0", "1990.0", "2012.0", "2003.0"));
    expect(labels(sorted)).toEqual(["1990.0", "2003.0", "2008.0", "2012.0"]);
  });

  it("falls back to alphabetical for non-ordinal dimensions (regions)", () => {
    const sorted = sortStratBars(bars("South East", "London", "Wales", "Scotland"));
    expect(labels(sorted)).toEqual(["London", "Scotland", "South East", "Wales"]);
  });

  it("does not mutate its input", () => {
    const input = bars("50-60%", "40-50%");
    const before = labels(input);
    sortStratBars(input);
    expect(labels(input)).toEqual(before);
  });
});
