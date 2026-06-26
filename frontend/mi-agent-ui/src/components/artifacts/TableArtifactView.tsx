import type { TableArtifact, TableColumn } from "@/domain";
import { cn, formatHeading, formatValue } from "@/lib/utils";

export function TableArtifactView({ artifact }: { artifact: TableArtifact }) {
  const maxByCol: Record<string, number> = {};
  for (const col of artifact.columns) {
    if (col.bar) {
      maxByCol[col.key] = Math.max(...artifact.rows.map((r) => Number(r[col.key]) || 0));
    }
  }

  const cell = (value: string | number, col: TableColumn) =>
    col.format === "number" && typeof value === "number"
      ? value.toLocaleString("en-GB")
      : formatValue(value, col.format, col.scale);

  return (
    <div className="overflow-hidden rounded-lg border border-[var(--color-line-soft)]">
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="bg-navy-850/80">
            {artifact.columns.map((col) => (
              <th
                key={col.key}
                className={cn(
                  "px-3 py-2 text-[11px] font-semibold uppercase tracking-wider text-ink-400",
                  col.align === "right" ? "text-right" : "text-left",
                )}
              >
                {formatHeading(col.label)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {artifact.rows.map((row, i) => (
            <tr key={i} className="border-t border-[var(--color-line-soft)] transition-colors hover:bg-navy-800/50">
              {artifact.columns.map((col) => {
                const raw = row[col.key];
                const pct = col.bar ? (Number(raw) / (maxByCol[col.key] || 1)) * 100 : 0;
                return (
                  <td
                    key={col.key}
                    className={cn(
                      "relative px-3 py-2 tabular-nums",
                      col.align === "right" ? "text-right" : "text-left",
                      col.format === "text" ? "font-medium text-ink-100" : "font-mono text-ink-300",
                    )}
                  >
                    {col.bar && (
                      <span className="absolute inset-y-1 left-0 -z-0 rounded-sm bg-peri-400/10" style={{ width: `${pct}%` }} aria-hidden />
                    )}
                    <span className="relative z-10">{cell(raw, col)}</span>
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
