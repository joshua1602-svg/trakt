import type { TableArtifactData, TableColumn } from "@/types";
import { cn } from "@/lib/utils";

function fmtCell(value: string | number, col: TableColumn): string {
  if (typeof value === "number") {
    if (col.format === "pct") return `${value.toFixed(1)}%`;
    if (col.format === "gbp") return `£${value.toLocaleString()}`;
    if (col.format === "number") return value.toLocaleString();
  }
  return String(value);
}

export function TableArtifact({ data }: { data: TableArtifactData }) {
  const maxByCol: Record<string, number> = {};
  for (const col of data.columns) {
    if (col.bar) {
      maxByCol[col.key] = Math.max(
        ...data.rows.map((r) => Number(r[col.key]) || 0),
      );
    }
  }

  return (
    <div className="overflow-hidden rounded-lg border border-[var(--color-line-soft)]">
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="bg-navy-850/80">
            {data.columns.map((col) => (
              <th
                key={col.key}
                className={cn(
                  "px-3 py-2 text-[11px] font-semibold uppercase tracking-wider text-ink-400",
                  col.align === "right" ? "text-right" : "text-left",
                )}
              >
                {col.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.rows.map((row, i) => (
            <tr
              key={i}
              className="border-t border-[var(--color-line-soft)] transition-colors hover:bg-navy-800/50"
            >
              {data.columns.map((col) => {
                const raw = row[col.key];
                const pct = col.bar
                  ? (Number(raw) / (maxByCol[col.key] || 1)) * 100
                  : 0;
                return (
                  <td
                    key={col.key}
                    className={cn(
                      "relative px-3 py-2 tabular-nums",
                      col.align === "right" ? "text-right" : "text-left",
                      col.format === "text"
                        ? "font-medium text-ink-100"
                        : "font-mono text-ink-300",
                    )}
                  >
                    {col.bar && (
                      <span
                        className="absolute inset-y-1 left-0 -z-0 rounded-sm bg-peri-400/10"
                        style={{ width: `${pct}%` }}
                        aria-hidden
                      />
                    )}
                    <span className="relative z-10">{fmtCell(raw, col)}</span>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
