import type { GeoArea, GeoExposure } from "@/domain/geo";
import atlasRaw from "@/data/geo/uk_itl3_paths.json";

const atlas = atlasRaw as { areas: Record<string, { name: string }> };

// ERM-plausible regional weighting (South / South-West heavy), by ITL1 prefix.
const REGION: Record<string, number> = {
  TLK: 1.0, TLJ: 0.85, TLI: 0.55, TLH: 0.5, TLG: 0.38, TLF: 0.36,
  TLL: 0.48, TLE: 0.3, TLD: 0.3, TLC: 0.26, TLM: 0.24, TLN: 0.18,
};
// A few forced hot areas (£m) so the map reads clearly.
const FORCED: Record<string, number> = {
  TLK51: 31.0, TLK52: 22.4, TLK63: 18.1, TLK30: 16.3, TLK43: 15.2,
  TLJ25: 13.8, TLK64: 12.6, TLK72: 10.4, TLK73: 9.7, TLK62: 9.1,
};

function hash(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return ((h >>> 0) % 1000) / 1000;
}

/** Deterministic mock exposure per ITL3 area, keyed to the real boundary atlas. */
export function mockGeoExposure(portfolioId: string): GeoExposure {
  const codes = Object.keys(atlas.areas);
  const raw: Array<{ code: string; balance: number }> = codes.map((code) => {
    const forced = FORCED[code];
    const weight = REGION[code.slice(0, 3)] ?? 0.25;
    // Non-forced areas spread £0.5m–£8m with a south/SW skew, so the whole map
    // gradients rather than flooring to the "no value" colour.
    const balance = forced != null
      ? forced * 1e6
      : Math.round(weight * (0.35 + 0.95 * hash(code)) * 6e6);
    return { code, balance };
  });
  const total = raw.reduce((a, r) => a + r.balance, 0);
  const areas: GeoArea[] = raw
    .map((r) => ({
      itl3_code: r.code,
      itl3_name: atlas.areas[r.code].name,
      balance: Math.round(r.balance * 100) / 100,
      count: Math.max(1, Math.round(r.balance / 320_000)),
      sharePct: total ? Math.round((r.balance / total) * 10000) / 100 : null,
    }))
    .sort((a, b) => b.balance - a.balance);

  return {
    dataset: "geo_itl3",
    portfolioId,
    available: true,
    areas,
    areaCount: areas.length,
    total: Math.round(total * 100) / 100,
    coveragePct: 96.3,
    resolvedFromItl3Field: 0,
    resolvedFromPostcode: areas.reduce((a, x) => a + x.count, 0),
    lookupAvailable: true,
    basis: "postcode_derived",
    currencyCode: "GBP",
  };
}
