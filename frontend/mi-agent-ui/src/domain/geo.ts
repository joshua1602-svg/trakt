/**
 * Geographic exposure by UK ITL3 area — the model behind the Geography tab's
 * choropleth. Mirrors `GET /mi/geo/exposure` (mi_agent_api/geo.py).
 */

/** One ITL3 area's funded exposure. */
export interface GeoArea {
  itl3_code: string;
  itl3_name: string;
  balance: number;
  count: number;
  sharePct: number | null;
}

export interface GeoExposure {
  dataset: "geo_itl3";
  portfolioId: string;
  available: boolean;
  /** Present when available === false. */
  reason?: string;
  areas: GeoArea[];
  areaCount?: number;
  total?: number;
  /** % of loans that resolved to an ITL3 area (postcode matched / field present). */
  coveragePct?: number;
  resolvedFromItl3Field?: number;
  resolvedFromPostcode?: number;
  lookupAvailable?: boolean;
  /** Where the ITL3 came from: "collateral" | "obligor" | "postcode_derived". */
  basis?: string;
  currencyCode?: string;
}

/** The bundled SVG-path atlas of UK ITL3 boundaries (src/data/geo/uk_itl3_paths.json). */
export interface ItlAtlas {
  viewBox: string;
  areas: Record<string, { name: string; d: string }>;
  note?: string;
}
