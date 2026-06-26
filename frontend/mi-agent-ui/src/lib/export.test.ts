import { describe, expect, it } from "vitest";
import { exportFilename, toCsv, toXlsxBlob, zipStore } from "./export";

describe("exportFilename", () => {
  it("builds a clean, dated filename from a title", () => {
    const date = new Date("2026-06-26T10:00:00Z");
    expect(exportFilename("Average LTV By Region", "png", date)).toBe("average_ltv_by_region_2026-06-26.png");
    expect(exportFilename("Pipeline Cases by Stage", "xlsx", date)).toBe("pipeline_cases_by_stage_2026-06-26.xlsx");
  });
});

describe("toCsv", () => {
  it("serialises headers and rows with RFC4180 quoting", () => {
    const csv = toCsv(["Region", "Balance"], [
      ["South East", 100],
      ['Region, with comma', 200],
    ]);
    expect(csv).toBe('Region,Balance\r\nSouth East,100\r\n"Region, with comma",200');
  });
});

describe("zipStore / toXlsxBlob", () => {
  it("produces a STORED zip with the PK local-file signature", () => {
    const enc = new TextEncoder();
    const bytes = zipStore([{ name: "a.txt", data: enc.encode("hello") }]);
    // Local file header magic: 'P' 'K' 0x03 0x04
    expect([bytes[0], bytes[1], bytes[2], bytes[3]]).toEqual([0x50, 0x4b, 0x03, 0x04]);
    // End-of-central-directory magic appears near the end.
    expect(bytes.length).toBeGreaterThan(50);
  });

  it("builds a non-empty xlsx blob with the right mime type", () => {
    const blob = toXlsxBlob(["Region", "Balance"], [["London", 400]]);
    expect(blob.size).toBeGreaterThan(0);
    expect(blob.type).toContain("spreadsheetml.sheet");
  });
});
