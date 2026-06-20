import { ResponsiveContainer, Tooltip, Treemap } from "recharts";
import type { ChartArtifact } from "@/domain";
import { THEME } from "@/lib/theme";
import { formatValue } from "@/lib/utils";

/**
 * Native treemap via Recharts' built-in `<Treemap>`, themed to Trakt. Renders a
 * top-level hierarchy (`xKey` label → `valueKey` size) without any Plotly dep.
 */

interface CellProps {
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  index?: number;
  name?: string;
}

function Cell({ x = 0, y = 0, width = 0, height = 0, index = 0, name = "" }: CellProps) {
  const fill = THEME.categorical[index % THEME.categorical.length];
  const showLabel = width > 56 && height > 24;
  return (
    <g>
      <rect x={x} y={y} width={width} height={height} fill={fill} stroke="#11162e" strokeWidth={2} rx={3} />
      {showLabel && (
        <text x={x + 6} y={y + 16} fill="#0c1024" fontSize={11} fontWeight={600} className="pointer-events-none">
          {name}
        </text>
      )}
    </g>
  );
}

export function TreemapArtifactView({ artifact }: { artifact: ChartArtifact }) {
  const { xKey, valueKey, rows, valueFormat } = artifact;
  if (!xKey || !valueKey) {
    return (
      <div className="rounded-lg border border-[var(--color-line-soft)] bg-navy-850/40 p-4 text-sm text-ink-400">
        Treemap data is incomplete (needs a dimension and a measure).
      </div>
    );
  }

  const data = rows.map((r) => ({ name: String(r[xKey]), size: Number(r[valueKey]) || 0 }));

  return (
    <ResponsiveContainer width="100%" height={320}>
      <Treemap data={data} dataKey="size" content={<Cell />} isAnimationActive={false}>
        <Tooltip
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null;
            const p = payload[0].payload as { name: string; size: number };
            return (
              <div className="rounded-lg border border-[var(--color-line)] bg-navy-950/95 px-3 py-2 text-xs shadow-xl">
                <div className="mb-0.5 font-medium text-ink-200">{p.name}</div>
                <div className="font-mono text-ink-100">{formatValue(p.size, valueFormat)}</div>
              </div>
            );
          }}
        />
      </Treemap>
    </ResponsiveContainer>
  );
}
