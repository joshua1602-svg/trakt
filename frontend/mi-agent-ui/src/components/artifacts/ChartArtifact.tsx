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
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ChartArtifactData } from "@/types";

const AXIS = "#6b7493";
const GRID = "#1c2440";

function valueFormatter(fmt?: ChartArtifactData["valueFormat"], unit?: string) {
  return (v: number) => {
    if (typeof v !== "number") return String(v);
    if (fmt === "gbp") return `£${v}${unit ?? ""}`;
    if (fmt === "pct") return `${v}%`;
    return v.toLocaleString();
  };
}

function ChartTooltip({
  active,
  payload,
  label,
  fmt,
}: {
  active?: boolean;
  payload?: Array<{ name: string; value: number; color: string }>;
  label?: string | number;
  fmt: (v: number) => string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-[var(--color-line)] bg-navy-950/95 px-3 py-2 text-xs shadow-xl">
      <div className="mb-1 font-medium text-ink-300">{label}</div>
      {payload.map((p) => (
        <div key={p.name} className="flex items-center gap-2">
          <span
            className="h-2 w-2 rounded-full"
            style={{ background: p.color }}
          />
          <span className="text-ink-400">{p.name}</span>
          <span className="ml-auto font-mono font-medium text-ink-100">
            {fmt(p.value)}
          </span>
        </div>
      ))}
    </div>
  );
}

/** Build cumulative offsets so stacked bars render as a waterfall. */
function toWaterfall(rows: ChartArtifactData["rows"]) {
  let cum = 0;
  return rows.map((r) => {
    const value = Number(r.value);
    const type = String(r.type);
    if (type === "total" || type === "base") {
      const out = { ...r, base: 0, bar: value, _kind: type };
      cum = value;
      return out;
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

const WF_COLORS: Record<string, string> = {
  base: "#3d4a82",
  add: "#919dd1",
  sub: "#e0607a",
  total: "#36c2a8",
};

function Legend({ data }: { data: ChartArtifactData }) {
  const items =
    data.chartType === "waterfall"
      ? [
          { label: "Base / total", color: "#3d4a82" },
          { label: "Inflow", color: "#919dd1" },
          { label: "Fallout", color: "#e0607a" },
          { label: "Forecast", color: "#36c2a8" },
        ]
      : data.series.map((s) => ({ label: s.label, color: s.color }));
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

function ChartBody({ data }: { data: ChartArtifactData }) {
  const fmt = valueFormatter(data.valueFormat, data.unit);
  const tickStyle = { fill: AXIS, fontSize: 11 };

  if (data.chartType === "waterfall") {
    const rows = toWaterfall(data.rows);
    return (
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={rows} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis
            dataKey={data.xKey}
            tick={tickStyle}
            axisLine={{ stroke: GRID }}
            tickLine={false}
            interval={0}
            angle={-18}
            textAnchor="end"
            height={56}
          />
          <YAxis
            tick={tickStyle}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v) => fmt(v)}
            width={56}
          />
          <Tooltip
            cursor={{ fill: "rgba(145,157,209,0.06)" }}
            content={({ active, payload, label }) => {
              if (!active || !payload?.length) return null;
              const row = payload[0].payload as Record<string, number>;
              return (
                <div className="rounded-lg border border-[var(--color-line)] bg-navy-950/95 px-3 py-2 text-xs shadow-xl">
                  <div className="mb-1 font-medium text-ink-300">{label}</div>
                  <div className="font-mono font-medium text-ink-100">
                    {fmt(Number(row.value))}
                  </div>
                </div>
              );
            }}
          />
          <Bar dataKey="base" stackId="w" fill="transparent" />
          <Bar dataKey="bar" stackId="w" radius={[3, 3, 0, 0]}>
            {rows.map((r, i) => (
              <Cell key={i} fill={WF_COLORS[String(r._kind)] ?? "#919dd1"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    );
  }

  if (data.chartType === "bar") {
    return (
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data.rows} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis
            dataKey={data.xKey}
            tick={tickStyle}
            axisLine={{ stroke: GRID }}
            tickLine={false}
            interval={0}
            angle={-18}
            textAnchor="end"
            height={56}
          />
          <YAxis
            tick={tickStyle}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v) => fmt(v)}
            width={56}
          />
          <Tooltip
            cursor={{ fill: "rgba(145,157,209,0.06)" }}
            content={(p) => (
              <ChartTooltip
                active={p.active}
                label={p.label}
                payload={
                  p.payload as unknown as {
                    name: string;
                    value: number;
                    color: string;
                  }[]
                }
                fmt={fmt}
              />
            )}
          />
          {data.series.map((s) => (
            <Bar key={s.key} dataKey={s.key} name={s.label} fill={s.color} radius={[3, 3, 0, 0]} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    );
  }

  if (data.chartType === "area") {
    return (
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={data.rows} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
          <defs>
            {data.series.map((s) => (
              <linearGradient key={s.key} id={`g-${s.key}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={s.color} stopOpacity={0.35} />
                <stop offset="100%" stopColor={s.color} stopOpacity={0} />
              </linearGradient>
            ))}
          </defs>
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis dataKey={data.xKey} tick={tickStyle} axisLine={{ stroke: GRID }} tickLine={false} />
          <YAxis tick={tickStyle} axisLine={false} tickLine={false} tickFormatter={(v) => fmt(v)} width={56} />
          <Tooltip
            content={(p) => (
              <ChartTooltip
                active={p.active}
                label={p.label}
                payload={
                  p.payload as unknown as {
                    name: string;
                    value: number;
                    color: string;
                  }[]
                }
                fmt={fmt}
              />
            )}
          />
          {data.series.map((s) => (
            <Area
              key={s.key}
              type="monotone"
              dataKey={s.key}
              name={s.label}
              stroke={s.color}
              strokeWidth={2}
              fill={`url(#g-${s.key})`}
            />
          ))}
        </AreaChart>
      </ResponsiveContainer>
    );
  }

  // line
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data.rows} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey={data.xKey} tick={tickStyle} axisLine={{ stroke: GRID }} tickLine={false} />
        <YAxis tick={tickStyle} axisLine={false} tickLine={false} tickFormatter={(v) => fmt(v)} width={56} />
        <Tooltip
            content={(p) => (
              <ChartTooltip
                active={p.active}
                label={p.label}
                payload={
                  p.payload as unknown as {
                    name: string;
                    value: number;
                    color: string;
                  }[]
                }
                fmt={fmt}
              />
            )}
          />
        {data.series.map((s) => (
          <Line
            key={s.key}
            type="monotone"
            dataKey={s.key}
            name={s.label}
            stroke={s.color}
            strokeWidth={2}
            dot={false}
            connectNulls
            activeDot={{ r: 4 }}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export function ChartArtifact({ data }: { data: ChartArtifactData }) {
  return (
    <div>
      <Legend data={data} />
      <ChartBody data={data} />
    </div>
  );
}
