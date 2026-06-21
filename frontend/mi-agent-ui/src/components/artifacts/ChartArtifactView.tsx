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
import type { ChartArtifact } from "@/domain";
import { THEME } from "@/lib/theme";
import { formatValue } from "@/lib/utils";

const AXIS = "#6b7493";
const GRID = "#1c2440";

type Fmt = ChartArtifact["valueFormat"];

function valueFormatter(fmt: Fmt, unit?: string) {
  return (v: number) => {
    if (typeof v !== "number") return String(v);
    if (fmt === "gbp") return `£${v}${unit ?? ""}`;
    if (fmt === "pct") return `${v}%`;
    return v.toLocaleString();
  };
}

/** Per-role formatter honouring a display scale (e.g. 0..1 LTV -> 51.0%). */
function roleFormatter(fmt: Fmt, scale?: number) {
  return (v: number) => (typeof v === "number" ? formatValue(v, fmt, scale) : String(v));
}

/** Controlled message when a chart artifact is missing a required role key,
 *  instead of silently rendering a blank chart. */
function ChartConfigError({ chartType, missing }: { chartType: string; missing: string[] }) {
  return (
    <div
      role="alert"
      className="rounded-lg border border-amber-500/40 bg-amber-500/5 px-3 py-4 text-xs text-amber-200"
    >
      Cannot render {chartType} chart: missing required key{missing.length > 1 ? "s" : ""}{" "}
      <span className="font-mono font-medium">{missing.join(", ")}</span>.
    </div>
  );
}

type RoleTooltipItem = { name: string; value: number; color: string };

/** Scatter/bubble tooltip: format each point by its role (x/y/size). */
function ScatterTooltip({
  active,
  payload,
  fmtByName,
}: {
  active?: boolean;
  payload?: RoleTooltipItem[];
  fmtByName: Record<string, (v: number) => string>;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-[var(--color-line)] bg-navy-950/95 px-3 py-2 text-xs shadow-xl">
      {payload.map((p) => (
        <div key={p.name} className="flex items-center gap-2">
          <span className="text-ink-400">{p.name}</span>
          <span className="ml-auto font-mono font-medium text-ink-100">
            {(fmtByName[p.name] ?? ((v: number) => String(v)))(p.value)}
          </span>
        </div>
      ))}
    </div>
  );
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
  const fmt = valueFormatter(artifact.valueFormat, artifact.unit);
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
    // Read roles by EXPLICIT key (never by series order). Missing a required key
    // is a controlled error, not a blank chart.
    const missing: string[] = [];
    if (!artifact.xKey) missing.push("xKey");
    if (!artifact.yKey) missing.push("yKey");
    if (artifact.chartType === "bubble" && !artifact.sizeKey) missing.push("sizeKey");
    if (missing.length) return <ChartConfigError chartType={artifact.chartType} missing={missing} />;

    const xLabel = artifact.xLabel ?? artifact.xKey;
    const yLabel = artifact.yLabel ?? artifact.yKey;
    const sizeLabel = artifact.sizeLabel ?? artifact.sizeKey;
    const xFmt = roleFormatter(artifact.xFormat, artifact.xScale);
    const yFmt = roleFormatter(artifact.yFormat, artifact.yScale);
    const sizeFmt = roleFormatter(artifact.sizeFormat, artifact.sizeScale);
    const fmtByName: Record<string, (v: number) => string> = {};
    if (xLabel) fmtByName[xLabel] = xFmt;
    if (yLabel) fmtByName[yLabel] = yFmt;
    if (sizeLabel) fmtByName[sizeLabel] = sizeFmt;

    return (
      <ResponsiveContainer width="100%" height={H}>
        <ScatterChart margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
          <CartesianGrid stroke={GRID} />
          <XAxis type="number" dataKey={artifact.xKey} name={xLabel} tick={tick} axisLine={{ stroke: GRID }} tickLine={false} tickFormatter={xFmt} />
          <YAxis type="number" dataKey={artifact.yKey} name={yLabel} tick={tick} axisLine={false} tickLine={false} width={56} tickFormatter={yFmt} />
          {artifact.chartType === "bubble" && artifact.sizeKey && (
            <ZAxis type="number" dataKey={artifact.sizeKey} name={sizeLabel} range={[40, 400]} />
          )}
          <Tooltip
            cursor={{ strokeDasharray: "3 3" }}
            content={(p) => (
              <ScatterTooltip active={p.active} payload={p.payload as unknown as RoleTooltipItem[]} fmtByName={fmtByName} />
            )}
          />
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
