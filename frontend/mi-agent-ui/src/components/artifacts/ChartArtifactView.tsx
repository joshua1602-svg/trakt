import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import type { ChartArtifact, DisplayHint } from "@/domain";
import { THEME } from "@/lib/theme";
import { toPercentPoints } from "@/lib/utils";

const AXIS = "#6b7493";
const GRID = "#1c2440";

type Fmt = ChartArtifact["valueFormat"];

/** A value formatter honouring the per-column display hint (format + scale). */
function hintFormatter(hint?: DisplayHint, fallbackFmt?: Fmt, unit?: string) {
  const fmt = hint?.format ?? fallbackFmt;
  const scale = hint?.scale;
  return (v: number) => {
    if (typeof v !== "number") return String(v);
    if (fmt === "gbp") return `£${v}${unit ?? ""}`;
    if (fmt === "pct") return `${toPercentPoints(v, scale).toFixed(1)}%`;
    return v.toLocaleString();
  };
}

type TooltipItem = { name: string; value: number; color: string };

function ChartTooltip({
  active,
  payload,
  label,
  fmt,
}: {
  active?: boolean;
  payload?: TooltipItem[];
  label?: string | number;
  fmt: (v: number) => string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-[var(--color-line)] bg-navy-950/95 px-3 py-2 text-xs shadow-xl">
      <div className="mb-1 font-medium text-ink-300">{label}</div>
      {payload.map((p) => (
        <div key={p.name} className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full" style={{ background: p.color }} />
          <span className="text-ink-400">{p.name}</span>
          <span className="ml-auto font-mono font-medium text-ink-100">{fmt(p.value)}</span>
        </div>
      ))}
    </div>
  );
}

const WF_COLORS: Record<string, string> = {
  base: "#3d4a82",
  add: THEME.peri,
  sub: THEME.negative,
  total: THEME.positive,
};

function toWaterfall(rows: ChartArtifact["rows"]) {
  let cum = 0;
  return rows.map((r) => {
    const value = Number(r.value);
    const type = String(r.type);
    if (type === "total" || type === "base") {
      cum = value;
      return { ...r, base: 0, bar: value, _kind: type };
    }
    if (value >= 0) {
      const out = { ...r, base: cum, bar: value, _kind: "add" };
      cum += value;
      return out;
    }
    const out = { ...r, base: cum + value, bar: -value, _kind: "sub" };
    cum += value;
    return out;
  });
}

function Legend({ artifact }: { artifact: ChartArtifact }) {
  const items =
    artifact.chartType === "waterfall"
      ? [
          { label: "Base / total", color: "#3d4a82" },
          { label: "Inflow", color: THEME.peri },
          { label: "Fallout", color: THEME.negative },
          { label: "Forecast", color: THEME.positive },
        ]
      : artifact.series.map((s) => ({ label: s.label, color: s.color }));
  if (items.length < 2) return null;
  return (
    <div className="mb-2 flex flex-wrap gap-x-4 gap-y-1">
      {items.map((i) => (
        <span key={i.label} className="inline-flex items-center gap-1.5 text-[11px] text-ink-400">
          <span className="h-2 w-2 rounded-full" style={{ background: i.color }} />
          {i.label}
        </span>
      ))}
    </div>
  );
}

const H = 300;

function Body({ artifact }: { artifact: ChartArtifact }) {
  // Format the value axis/tooltip from the value column's display hint (so a
  // fraction-stored percent renders as points), falling back to valueFormat.
  const valueCol = artifact.series[0]?.key ?? artifact.valueKey;
  const valueHint = valueCol ? artifact.displayHints?.[valueCol] : undefined;
  const fmt = hintFormatter(valueHint, artifact.valueFormat, artifact.unit);
  const tick = { fill: AXIS, fontSize: 11 };
  const yAxis = (
    <YAxis tick={tick} axisLine={false} tickLine={false} tickFormatter={(v) => fmt(v)} width={56} />
  );
  const tip = (
    <Tooltip
      cursor={{ fill: "rgba(145,157,209,0.06)" }}
      content={(p) => (
        <ChartTooltip active={p.active} label={p.label} payload={p.payload as unknown as TooltipItem[]} fmt={fmt} />
      )}
    />
  );

  if (artifact.chartType === "waterfall") {
    const rows = toWaterfall(artifact.rows);
    return (
      <ResponsiveContainer width="100%" height={H}>
        <BarChart data={rows} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis dataKey={artifact.xKey} tick={tick} axisLine={{ stroke: GRID }} tickLine={false} interval={0} angle={-18} textAnchor="end" height={56} />
          {yAxis}
          {tip}
          <Bar dataKey="base" stackId="w" fill="transparent" />
          <Bar dataKey="bar" stackId="w" radius={[3, 3, 0, 0]}>
            {rows.map((r, i) => (
              <Cell key={i} fill={WF_COLORS[String(r._kind)] ?? THEME.peri} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    );
  }

  if (artifact.chartType === "bar") {
    return (
      <ResponsiveContainer width="100%" height={H}>
        <BarChart data={artifact.rows} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis dataKey={artifact.xKey} tick={tick} axisLine={{ stroke: GRID }} tickLine={false} interval={0} angle={-18} textAnchor="end" height={56} />
          {yAxis}
          {tip}
          {artifact.series.map((s) => (
            <Bar key={s.key} dataKey={s.key} name={s.label} fill={s.color} radius={[3, 3, 0, 0]} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    );
  }

  if (artifact.chartType === "area") {
    return (
      <ResponsiveContainer width="100%" height={H}>
        <AreaChart data={artifact.rows} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
          <defs>
            {artifact.series.map((s) => (
              <linearGradient key={s.key} id={`g-${artifact.id}-${s.key}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={s.color} stopOpacity={0.35} />
                <stop offset="100%" stopColor={s.color} stopOpacity={0} />
              </linearGradient>
            ))}
          </defs>
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis dataKey={artifact.xKey} tick={tick} axisLine={{ stroke: GRID }} tickLine={false} />
          {yAxis}
          {tip}
          {artifact.series.map((s) => (
            <Area key={s.key} type="monotone" dataKey={s.key} name={s.label} stroke={s.color} strokeWidth={2} fill={`url(#g-${artifact.id}-${s.key})`} />
          ))}
        </AreaChart>
      </ResponsiveContainer>
    );
  }

  if (artifact.chartType === "scatter" || artifact.chartType === "bubble") {
    // Consume EXPLICIT role keys from the API; fall back to series order only
    // for older payloads. The y axis is never inferred-then-null.
    const hints = artifact.displayHints ?? {};
    const xKey = artifact.xKey ?? artifact.series[0]?.key;
    const yKey = artifact.yKey ?? artifact.series[1]?.key;
    const sizeKey = artifact.sizeKey ?? artifact.series[2]?.key;
    const xLabel = artifact.xLabel ?? artifact.series[0]?.label;
    const yLabel = artifact.yLabel ?? artifact.series[1]?.label;
    const xFmt = hintFormatter(xKey ? hints[xKey] : undefined);
    const yFmt = hintFormatter(yKey ? hints[yKey] : undefined);
    return (
      <ResponsiveContainer width="100%" height={H}>
        <ScatterChart margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
          <CartesianGrid stroke={GRID} />
          <XAxis type="number" dataKey={xKey} name={xLabel} tick={tick} axisLine={{ stroke: GRID }} tickLine={false} tickFormatter={(v) => xFmt(v)} />
          <YAxis type="number" dataKey={yKey} name={yLabel} tick={tick} axisLine={false} tickLine={false} width={56} tickFormatter={(v) => yFmt(v)} />
          {artifact.chartType === "bubble" && sizeKey && <ZAxis type="number" dataKey={sizeKey} range={[40, 400]} />}
          {tip}
          <Scatter data={artifact.rows} fill={THEME.peri} fillOpacity={0.6} />
        </ScatterChart>
      </ResponsiveContainer>
    );
  }

  // line
  return (
    <ResponsiveContainer width="100%" height={H}>
      <LineChart data={artifact.rows} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey={artifact.xKey} tick={tick} axisLine={{ stroke: GRID }} tickLine={false} />
        {yAxis}
        {tip}
        {artifact.series.map((s) => (
          <Line key={s.key} type="monotone" dataKey={s.key} name={s.label} stroke={s.color} strokeWidth={2} dot={false} connectNulls activeDot={{ r: 4 }} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export function ChartArtifactView({ artifact }: { artifact: ChartArtifact }) {
  return (
    <div>
      <Legend artifact={artifact} />
      <Body artifact={artifact} />
    </div>
  );
}
