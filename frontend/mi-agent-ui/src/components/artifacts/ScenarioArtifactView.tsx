import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ScenarioArtifact } from "@/domain";
import { THEME } from "@/lib/theme";
import { formatGBP } from "@/lib/utils";

const AXIS = "#6b7493";
const GRID = "#1c2440";

export function ScenarioArtifactView({ artifact }: { artifact: ScenarioArtifact }) {
  const tick = { fill: AXIS, fontSize: 11 };
  return (
    <div>
      <div className="mb-3 flex flex-wrap gap-1.5">
        {Object.entries(artifact.assumptions).map(([k, v]) => (
          <span key={k} className="inline-flex items-center gap-1 rounded-md border border-[var(--color-line-soft)] bg-navy-850/60 px-2 py-1 text-[10px] text-ink-400">
            <span className="text-ink-500">{k}</span>
            <span className="font-mono font-medium text-ink-200">{v}</span>
          </span>
        ))}
      </div>

      <div className="mb-2 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-ink-400">
        <span className="inline-flex items-center gap-1.5"><span className="h-2 w-2 rounded-full" style={{ background: THEME.peri }} />Balance (£MM)</span>
        <span className="inline-flex items-center gap-1.5"><span className="h-2 w-2 rounded-full" style={{ background: THEME.positive }} />Portfolio LTV (%)</span>
        <span className="inline-flex items-center gap-1.5"><span className="h-2 w-2 rounded-full" style={{ background: THEME.negative }} />Cumulative NNEG (£MM)</span>
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={artifact.projection} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis dataKey="year" tick={tick} axisLine={{ stroke: GRID }} tickLine={false} tickFormatter={(v) => `Y${v}`} />
          <YAxis yAxisId="left" tick={tick} axisLine={false} tickLine={false} width={52} tickFormatter={(v) => `£${v}`} />
          <YAxis yAxisId="right" orientation="right" tick={tick} axisLine={false} tickLine={false} width={40} tickFormatter={(v) => `${v}%`} />
          <Tooltip
            cursor={{ fill: "rgba(145,157,209,0.06)" }}
            content={({ active, payload, label }) => {
              if (!active || !payload?.length) return null;
              const row = payload[0].payload as ScenarioArtifact["projection"][number];
              return (
                <div className="rounded-lg border border-[var(--color-line)] bg-navy-950/95 px-3 py-2 text-xs shadow-xl">
                  <div className="mb-1 font-medium text-ink-300">Year {label}</div>
                  <div className="space-y-0.5 font-mono text-ink-100">
                    <div>Balance {formatGBP(row.balance * 1e6)}</div>
                    <div>LTV {row.ltv.toFixed(1)}%</div>
                    <div>Cum. NNEG {formatGBP(row.cumulativeNneg * 1e6)}</div>
                  </div>
                </div>
              );
            }}
          />
          <Bar yAxisId="left" dataKey="balance" name="Balance" fill={THEME.peri} fillOpacity={0.5} radius={[3, 3, 0, 0]} />
          <Line yAxisId="right" type="monotone" dataKey="ltv" name="LTV" stroke={THEME.positive} strokeWidth={2} dot={false} />
          <Line yAxisId="left" type="monotone" dataKey="cumulativeNneg" name="Cumulative NNEG" stroke={THEME.negative} strokeWidth={2} dot={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
