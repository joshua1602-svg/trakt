import { useEffect, useMemo, useState } from "react";
import type { AgentClient } from "@/api/AgentClient";
import type { GeoExposure, ItlAtlas } from "@/domain/geo";
import atlasRaw from "@/data/geo/uk_itl3_paths.json";
import { UkChoropleth, type AreaDetail } from "@/components/UkChoropleth";
import { formatGBP } from "@/lib/utils";

const atlas = atlasRaw as unknown as ItlAtlas;

const money = (v: number) => formatGBP(v, { compact: true });

/** Heat ramp preview matching UkChoropleth (cool low → warm high). */
function Legend() {
  return (
    <div className="mt-1.5 flex items-center gap-2 text-[11px] text-ink-500">
      <span>Lower</span>
      <span className="h-2 flex-1 rounded"
        style={{ background: "linear-gradient(90deg,#1a2340,#2f4a94,#4f74d6,#37b9a6,#e8b13c)" }} />
      <span>Higher</span>
    </div>
  );
}

function Kpi({ label, value, detail }: { label: string; value: React.ReactNode; detail?: string }) {
  return (
    <div className="rounded-xl border border-[var(--color-line-soft)] bg-navy-900/40 px-3.5 py-3">
      <div className="text-[11px] uppercase tracking-wide text-ink-500">{label}</div>
      <div className="mt-0.5 font-display text-[1.35rem] font-bold tabular-nums tracking-tight text-ink-100">{value}</div>
      {detail && <div className="mt-0.5 text-[12px] text-ink-400">{detail}</div>}
    </div>
  );
}

/** Core Dashboard → Geography: funded exposure concentration across UK ITL3
 * areas as a choropleth + ranked list, from GET /mi/geo/exposure. */
export function GeographyPanel({ client, portfolioId }: {
  client: AgentClient; portfolioId: string;
}) {
  const [geo, setGeo] = useState<GeoExposure | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    Promise.resolve(client.getGeoExposure(portfolioId))
      .then((r) => { if (!cancelled) setGeo(r); })
      .catch(() => {})
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [client, portfolioId]);

  const { valueByCode, shareByCode, detailByCode, ranked, total, top, top5Pct } = useMemo(() => {
    const areas = geo?.areas ?? [];
    const v: Record<string, number> = {};
    const s: Record<string, number | null> = {};
    const d: Record<string, AreaDetail> = {};
    for (const a of areas) {
      v[a.itl3_code] = a.balance;
      s[a.itl3_code] = a.sharePct;
      d[a.itl3_code] = { avgTicket: a.avgTicket, avgLtv: a.avgLtv, avgAge: a.avgAge };
    }
    const t = geo?.total ?? areas.reduce((acc, a) => acc + a.balance, 0);
    const sorted = [...areas].sort((a, b) => b.balance - a.balance);
    const top5 = sorted.slice(0, 5).reduce((acc, a) => acc + a.balance, 0);
    return {
      valueByCode: v, shareByCode: s, detailByCode: d, ranked: sorted, total: t,
      top: sorted[0], top5Pct: t ? (top5 / t) * 100 : 0,
    };
  }, [geo]);

  if (loading && !geo) {
    return <p className="text-[12px] text-ink-500" data-testid="geo-loading">Loading geographic exposure…</p>;
  }

  if (geo && !geo.available) {
    return (
      <div className="rounded-lg border border-amber-400/20 bg-amber-400/5 px-3 py-2 text-[11px] text-amber-300/90"
        data-testid="geo-unavailable">
        No geographic exposure for this run{geo.reason ? ` — ${geo.reason}` : ""}.
        A property postcode or ITL3 region is needed on the funded tape.
      </div>
    );
  }

  const maxBalance = ranked[0]?.balance ?? 0;

  return (
    <div className="space-y-3" data-testid="geography-view">
      <div className="grid grid-cols-2 gap-2.5 lg:grid-cols-4">
        <Kpi label="Total funded exposure" value={money(total)}
          detail={`${geo?.areaCount ?? ranked.length} ITL3 areas`} />
        <Kpi label="Top area"
          value={<span className="text-amber-300">{top ? top.itl3_name.split(",")[0].split(" (")[0] : "—"}</span>}
          detail={top ? `${money(top.balance)} · ${(top.sharePct ?? 0).toFixed(1)}% of book` : undefined} />
        <Kpi label="Top-5 concentration" value={`${top5Pct.toFixed(1)}%`} detail="of funded exposure" />
        <Kpi label="Postcode coverage" value={`${(geo?.coveragePct ?? 0).toFixed(1)}%`}
          detail="loans resolved to an area" />
      </div>

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-[1.15fr_0.85fr]">
        <div className="rounded-xl border border-[var(--color-line)] bg-navy-900/40 p-4">
          <div className="text-[12px] font-semibold text-ink-200">Exposure map</div>
          <p className="mb-2 mt-0.5 text-[11px] text-ink-500">
            Each ITL3 area shaded by funded exposure — brighter = higher. Hover for exposure,
            average ticket, LTV and borrower age.
          </p>
          <UkChoropleth atlas={atlas} valueByCode={valueByCode} shareByCode={shareByCode}
            detailByCode={detailByCode} formatValue={money} formatTicket={money}
            className="max-w-[440px]" />
          <Legend />
        </div>

        <div className="rounded-xl border border-[var(--color-line)] bg-navy-900/40 p-4">
          <div className="text-[12px] font-semibold text-ink-200">Top 15 areas</div>
          <p className="mb-2 mt-0.5 text-[11px] text-ink-500">Ranked by funded exposure.</p>
          <ol className="flex flex-col gap-1" data-testid="geo-rank">
            {ranked.slice(0, 15).map((a, i) => (
              <li key={a.itl3_code}
                className="grid grid-cols-[1.1rem_1fr_auto] items-center gap-2 text-[12px]">
                <span className="text-right font-mono text-[11px] text-ink-500">{i + 1}</span>
                <span className="relative h-6 overflow-hidden rounded border border-[var(--color-line-soft)] bg-navy-900/60">
                  <span className="absolute inset-y-0 left-0 rounded-l"
                    style={{
                      width: `${maxBalance ? (a.balance / maxBalance) * 100 : 0}%`,
                      background: "linear-gradient(90deg,rgba(79,116,214,.4),#37b9a6)",
                    }} />
                  <span className="absolute inset-y-0 left-2 flex items-center whitespace-nowrap text-[11px] font-medium text-ink-100"
                    style={{ textShadow: "0 1px 2px rgba(0,0,0,.45)" }}>
                    {a.itl3_name}
                  </span>
                </span>
                <span className="min-w-[3.3rem] text-right font-mono tabular-nums text-[11px] text-ink-200">
                  {money(a.balance)}
                </span>
              </li>
            ))}
          </ol>
        </div>
      </div>

      <p className="text-[10px] text-ink-500" data-testid="geo-lineage">
        ITL3 area from each loan's {geo?.basis === "postcode_derived" ? "property postcode (master lookup)" : "tape ITL3 field"}.
        Boundaries: ONS ITL3 (Jan 2025), simplified. {geo?.currencyCode && geo.currencyCode !== "GBP" ? `Currency: ${geo.currencyCode}.` : ""}
      </p>
    </div>
  );
}
